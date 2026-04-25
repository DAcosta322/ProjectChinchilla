"""Round 3 FINAL - consolidated from research/r1..r10 sweep.

Inherits all of eff4.py (MR + VEV4/4.5 spillover + VEV5000 adaptive
TV + profit-taking) and adds:

  + HYD-only vol-adaptive MR_STRENGTH  (r8)   -> +~$3.8K/4-day
  + Adverse-selection (toxicity) damper (r5)  -> ~tied, defensive

Discarded (worse or marginal): order-book imbalance, trade-tape flow,
HYD/VEL pairs (no signal), Kalman anchor (mistuned for this data),
layered passive quoting (no fill benefit), L1 lag1 reversion bias,
IV-based OTM-voucher pricing (TV too unstable), cross-strike arb
(book parity enforced at all observed times).

Noise-stability hardening:
  - All EWMA / variance estimates have safe initialization and
    clipping to avoid blow-up on early ticks
  - vol_factor and toxicity are clipped to bounded ranges
  - No SAR / position-flipping logic anywhere - smooth target only
  - Profit-taking decay is monotone (one-direction) to avoid
    oscillation at threshold
  - Quote prices clipped against fv_eff to never post worse than
    our valuation
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json
import math


# ---------------------------------------------------------------------------
# Per-product params
# ---------------------------------------------------------------------------
class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    ANCHOR_SPAN = 500
    MR_STRENGTH = 5
    PROFIT_DIST = 40
    PROFIT_RANGE = 40
    # vol-scaling on MR_STRENGTH (r8): ref std of ~2 ticks/tick
    VOL_SPAN = 200
    VOL_REF = 2.0
    # toxicity damper on skew (r5). Code wired in but disabled by
    # default (TOX_FULL_OFF=0). Backtest data shows it costs
    # ~$1-4K because the simulated flow isn't adverse. Set to a
    # finite value (e.g. 20) live if the live market is more hostile.
    TOX_SPAN = 50
    TOX_FULL_OFF = 0
    # Cont-Stoikov order flow imbalance (t2) - +$3.8K backtest
    OFI_GAIN = 1.0
    OFI_SPAN = 20


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    PROFIT_DIST = 15
    PROFIT_RANGE = 15
    VOL_SPAN = 0          # disabled - VELVET MR works better at fixed gain
    VOL_REF = 1.0
    TOX_SPAN = 50
    TOX_FULL_OFF = 0
    OFI_GAIN = 1.0
    OFI_SPAN = 20


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    VOL_SPAN = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0
    # Own short MR on top of spillover - small gentle anchor (span=1000)
    # contributes ~$300-500 to 4-day total without affecting day 3.
    OWN_ANCHOR = 1000
    OWN_MR = 5


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    VOL_SPAN = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0
    OWN_ANCHOR = 1000
    OWN_MR = 5


class VEV5000Params:
    SYMBOL = "VEV_5000"
    STRIKE = 5000
    POS_LIMIT = 300
    INV_SKEW = 2
    TV_EWMA_SPAN = 50       # tuned via sensitivity sweep - faster tracking
    MIN_SAMPLES = 100
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    VOL_SPAN = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _micro_mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    bv = od.buy_orders[bb]
    av = -od.sell_orders[ba]
    if bv + av <= 0:
        return (bb + ba) / 2.0
    return (bb * av + ba * bv) / (bv + av)


def _update_avg_entry(prev_avg: float, prev_pos: int, own_trades: list) -> float:
    cur_pos = prev_pos
    avg = prev_avg if prev_pos != 0 else 0.0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION":
            qty = t.quantity
        elif getattr(t, "seller", None) == "SUBMISSION":
            qty = -t.quantity
        else:
            continue
        new_pos = cur_pos + qty
        if new_pos == 0:
            avg = 0.0
        elif cur_pos == 0 or (cur_pos > 0) != (new_pos > 0):
            avg = float(t.price)
        elif (cur_pos > 0) == (qty > 0):
            avg = (avg * cur_pos + t.price * qty) / new_pos
        cur_pos = new_pos
    return avg


def _profit_decay(P, pos: int, fv: float, avg_entry: float) -> float:
    if avg_entry <= 0 or pos == 0 or P.PROFIT_DIST <= 0:
        return 1.0
    sign = 1.0 if pos > 0 else -1.0
    profit_per_unit = (fv - avg_entry) * sign
    if profit_per_unit <= P.PROFIT_DIST:
        return 1.0
    excess = profit_per_unit - P.PROFIT_DIST
    return max(0.0, 1.0 - excess / max(1.0, P.PROFIT_RANGE))


def _book_top(od: OrderDepth):
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, od.buy_orders[bb], ba, -od.sell_orders[ba]


def _ofi_step(prev, cur):
    """Cont-Stoikov OFI for one tick. Returns net buy-side flow."""
    if prev is None or cur is None:
        return 0.0
    bb1, bv1, ba1, av1 = prev
    bb2, bv2, ba2, av2 = cur
    if bb2 > bb1:
        e_bid = bv2
    elif bb2 < bb1:
        e_bid = -bv1
    else:
        e_bid = bv2 - bv1
    if ba2 > ba1:
        e_ask = av1
    elif ba2 < ba1:
        e_ask = -av2
    else:
        e_ask = av1 - av2
    return float(e_bid + e_ask)


def _update_toxicity(P, prev_tox: float, own_trades: list, fv_now: float) -> float:
    if not own_trades or P.TOX_SPAN <= 0:
        return prev_tox * 0.95
    bad = 0.0
    n = 0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION":
            adv = max(0.0, t.price - fv_now)
        elif getattr(t, "seller", None) == "SUBMISSION":
            adv = max(0.0, fv_now - t.price)
        else:
            continue
        bad += adv
        n += 1
    if n == 0:
        return prev_tox * 0.95
    avg_adverse = bad / n
    a = 2.0 / (P.TOX_SPAN + 1.0)
    return a * avg_adverse + (1 - a) * prev_tox


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}
        ema = td.get("ema", {})
        avg_entry = td.get("avg", {})
        tv = td.get("tv", {})
        tv_n = td.get("tvn", {})
        prev_fv = td.get("pfv", {})
        var_ewma = td.get("var", {})
        tox = td.get("tox", {})
        prev_book = td.get("pbook", {})
        ofi_ewma = td.get("ofi", {})

        # Refresh avg_entry from own_trades for products that need it
        for P in (HydrogelParams, VelvetParams):
            if P.PROFIT_DIST <= 0 and P.TOX_FULL_OFF <= 0:
                continue
            sym = P.SYMBOL
            own = state.own_trades.get(sym, []) if state.own_trades else []
            cur_pos = state.position.get(sym, 0)
            delta = sum(t.quantity if getattr(t, "buyer", None) == "SUBMISSION"
                       else -t.quantity if getattr(t, "seller", None) == "SUBMISSION" else 0
                       for t in own)
            avg_entry[sym] = _update_avg_entry(
                avg_entry.get(sym, 0.0), cur_pos - delta, own)

        # HYDROGEL
        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                own = state.own_trades.get(HydrogelParams.SYMBOL, []) if state.own_trades else []
                tox[HydrogelParams.SYMBOL] = _update_toxicity(
                    HydrogelParams, tox.get(HydrogelParams.SYMBOL, 0.0), own, fv)
                _, orders = self._mr(state, HydrogelParams, fv, ema,
                                     avg_entry, prev_fv, var_ewma, tox,
                                     prev_book, ofi_ewma)
                result[HydrogelParams.SYMBOL] = orders

        # VELVET
        velvet_mid: Optional[float] = None
        velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                own = state.own_trades.get(VelvetParams.SYMBOL, []) if state.own_trades else []
                tox[VelvetParams.SYMBOL] = _update_toxicity(
                    VelvetParams, tox.get(VelvetParams.SYMBOL, 0.0), own, velvet_mid)
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema,
                                      avg_entry, prev_fv, var_ewma, tox,
                                      prev_book, ofi_ewma)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        # VEV_4000 / VEV_4500 spillover + VEV_5000 adaptive TV
        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    target = int(round(velvet_target_rel * P.POS_LIMIT))
                    # VEV own MR on top of spillover - small contribution
                    if getattr(P, 'OWN_MR', 0) > 0 and getattr(P, 'OWN_ANCHOR', 0) > 0:
                        prev = ema.get(P.SYMBOL)
                        if prev is None:
                            anchor = fv
                        else:
                            a = 2.0 / (P.OWN_ANCHOR + 1.0)
                            anchor = a * fv + (1 - a) * prev
                        ema[P.SYMBOL] = anchor
                        target += int(round(-P.OWN_MR * (fv - anchor)))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                    result[P.SYMBOL] = self._exec(state, P, fv, target)

            if VEV5000Params.SYMBOL in state.order_depths:
                od5 = state.order_depths[VEV5000Params.SYMBOL]
                vev5_mid = _micro_mid(od5)
                if vev5_mid is not None:
                    intrinsic = velvet_mid - VEV5000Params.STRIKE
                    observed = vev5_mid - intrinsic
                    prev = tv.get(VEV5000Params.SYMBOL)
                    a = 2.0 / (VEV5000Params.TV_EWMA_SPAN + 1.0)
                    cur = observed if prev is None else (
                        a * observed + (1 - a) * prev)
                    # Clip TV estimate to a sane band so a single bad
                    # tick can't poison the EWMA permanently.
                    cur = max(-50.0, min(50.0, cur))
                    tv[VEV5000Params.SYMBOL] = cur
                    n = tv_n.get(VEV5000Params.SYMBOL, 0) + 1
                    tv_n[VEV5000Params.SYMBOL] = n
                    if n >= VEV5000Params.MIN_SAMPLES:
                        result[VEV5000Params.SYMBOL] = self._exec(
                            state, VEV5000Params, intrinsic + cur, 0)

        return result, 0, json.dumps({
            "ema": ema, "avg": avg_entry, "tv": tv, "tvn": tv_n,
            "pfv": prev_fv, "var": var_ewma, "tox": tox,
            "pbook": prev_book, "ofi": ofi_ewma})

    def _mr(self, state: TradingState, P, fv: float,
            ema: Dict[str, float],
            avg_entry: Dict[str, float],
            prev_fv: Dict[str, float],
            var_ewma: Dict[str, float],
            tox: Dict[str, float],
            prev_book: Dict[str, Any],
            ofi_ewma: Dict[str, float]):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return 0, []

        # Volatility-adaptive MR strength (r8) - HYD only
        vol_factor = 1.0
        if P.VOL_SPAN > 0:
            if P.SYMBOL in prev_fv:
                ret = fv - prev_fv[P.SYMBOL]
                # Clip extreme returns so the var estimate isn't blown up
                # by a single anomalous jump.
                ret = max(-50.0, min(50.0, ret))
                prev_var = var_ewma.get(P.SYMBOL, P.VOL_REF * P.VOL_REF)
                a = 2.0 / (P.VOL_SPAN + 1.0)
                var_ewma[P.SYMBOL] = a * (ret * ret) + (1 - a) * prev_var
            prev_fv[P.SYMBOL] = fv
            std_est = math.sqrt(max(0.01, var_ewma.get(P.SYMBOL, 1.0)))
            vol_factor = max(0.5, min(2.0, P.VOL_REF / std_est))

        # Cont-Stoikov order flow imbalance (t2)
        ofi_smooth = 0.0
        if getattr(P, 'OFI_GAIN', 0) > 0 and getattr(P, 'OFI_SPAN', 0) > 0:
            cur_book = _book_top(od)
            ofi_one = _ofi_step(prev_book.get(P.SYMBOL), cur_book)
            prev_book[P.SYMBOL] = cur_book
            a_ofi = 2.0 / (P.OFI_SPAN + 1.0)
            ofi_smooth = a_ofi * ofi_one + (1 - a_ofi) * ofi_ewma.get(P.SYMBOL, 0.0)
            ofi_ewma[P.SYMBOL] = ofi_smooth

        target = 0
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            if prev is None:
                anchor = fv
            else:
                alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
                anchor = alpha * fv + (1 - alpha) * prev
            ema[P.SYMBOL] = anchor
            mr_eff = P.MR_STRENGTH * vol_factor
            raw = -mr_eff * (fv - anchor) + getattr(P, 'OFI_GAIN', 0) * ofi_smooth
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))

        decay = _profit_decay(P, pos, fv, avg_entry.get(P.SYMBOL, 0.0))
        target = int(round(target * decay))

        # Toxicity damper: shrink effective skew when adverse-flow EWMA
        # is high. Less crossing -> less paying spread on toxic flow.
        tox_factor = 1.0
        if P.TOX_FULL_OFF > 0:
            tox_factor = max(0.2, 1.0 - tox.get(P.SYMBOL, 0.0) / P.TOX_FULL_OFF)

        return target, self._do_orders(P, fv, pos, target, od, tox_factor)

    def _exec(self, state: TradingState, P, fv: float, target: int) -> List[Order]:
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return []
        return self._do_orders(P, fv, pos, target, od, 1.0)

    def _do_orders(self, P, fv: float, pos: int, target_pos: int,
                   od: OrderDepth, skew_factor: float = 1.0) -> List[Order]:
        orders: List[Order] = []
        skew = P.INV_SKEW * skew_factor * (pos - target_pos) / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        for price in sorted(od.sell_orders.keys()):
            if price < fv_eff and buy_cap > 0:
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fv_eff and sell_cap > 0:
                qty = min(od.buy_orders[price], sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty))
                sell_cap -= qty

        base_bid = best_bid + 1
        base_ask = best_ask - 1
        if base_bid >= base_ask:
            base_bid = best_bid
            base_ask = best_ask
        shift = int(round(-skew))
        our_bid = min(base_bid + shift, int(fv_eff))
        our_ask = max(base_ask + shift, int(fv_eff) + 1)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))
        return orders
