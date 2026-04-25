"""Round 3 EFFICIENCY v4 - eff3.py + PROFIT-TAKING.

Motivation: 386778 (trend-follow) hit a peak P&L of +$13.7K on day 3
HYDROGEL but gave back $9K on the bounce, finishing at +$4.8K. Even
our pure MR variants captured the bounce-side profit but not the
mid-day extreme. Profit-taking on a winning position scales the
target toward 0 once unrealized gain exceeds a threshold, banking
the trend portion of the move before mean reversion erases it.

Mechanism:
  Track avg_entry per symbol from state.own_trades (same as sl.py).
  Compute profit_per_unit = (fv - avg_entry) * sign(pos).
  If profit_per_unit > PROFIT_DIST, decay the MR target linearly:
    decay = max(0, 1 - (profit_per_unit - PROFIT_DIST) / PROFIT_RANGE)
    target_pos *= decay
  When fv has moved PROFIT_DIST + PROFIT_RANGE in our favor, target
  is fully 0 -> position closes. INV_SKEW execution unwinds passively
  rather than crossing the book hard.

Different from stop-loss: SL closes losers; this closes winners. The
core insight is that on a partial mean-reversion (price doesn't
return all the way to anchor) we maximize realized P&L by booking
mid-bounce instead of waiting for full reversion that may not happen.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    ANCHOR_SPAN = 500
    MR_STRENGTH = 5
    PROFIT_DIST = 40        # start scaling target down at this gain/unit
    PROFIT_RANGE = 40       # fully closed at PROFIT_DIST + PROFIT_RANGE


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    PROFIT_DIST = 15        # tighter band for VELVET (smaller range)
    PROFIT_RANGE = 15


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True
    PROFIT_DIST = 0         # off - VEVs use spillover, not own MR signal
    PROFIT_RANGE = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3
    SPILLOVER = True
    PROFIT_DIST = 0
    PROFIT_RANGE = 0


class VEV5000Params:
    SYMBOL = "VEV_5000"
    STRIKE = 5000
    POS_LIMIT = 300
    INV_SKEW = 2
    TV_EWMA_SPAN = 100
    MIN_SAMPLES = 100
    PROFIT_DIST = 0
    PROFIT_RANGE = 0


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


def _update_avg_entry(prev_avg: float, prev_pos: int, own_trades: list):
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
        ema: Dict[str, float] = td.get("ema", {})
        avg_entry: Dict[str, float] = td.get("avg", {})
        tv: Dict[str, float] = td.get("tv", {})
        tv_n: Dict[str, int] = td.get("tvn", {})

        # Refresh avg_entry from own_trades for products that use profit-take
        for P in (HydrogelParams, VelvetParams):
            if P.PROFIT_DIST <= 0:
                continue
            sym = P.SYMBOL
            own = state.own_trades.get(sym, []) if state.own_trades else []
            cur_pos = state.position.get(sym, 0)
            delta = 0
            for t in own:
                if getattr(t, "buyer", None) == "SUBMISSION":
                    delta += t.quantity
                elif getattr(t, "seller", None) == "SUBMISSION":
                    delta -= t.quantity
            pre_pos = cur_pos - delta
            prev_avg = avg_entry.get(sym, 0.0)
            avg_entry[sym] = _update_avg_entry(prev_avg, pre_pos, own)

        # HYDROGEL
        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                _, orders = self._mr(state, HydrogelParams, fv, ema, avg_entry)
                result[HydrogelParams.SYMBOL] = orders

        # VELVET (capture spillover target)
        velvet_mid: Optional[float] = None
        velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema, avg_entry)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        # VEV_4000 / VEV_4500
        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    target = int(round(velvet_target_rel * P.POS_LIMIT))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                    result[P.SYMBOL] = self._exec(state, P, fv, target)

            # VEV_5000 with adaptive TV
            if VEV5000Params.SYMBOL in state.order_depths:
                od5 = state.order_depths[VEV5000Params.SYMBOL]
                vev5_mid = _micro_mid(od5)
                if vev5_mid is not None:
                    intrinsic = velvet_mid - VEV5000Params.STRIKE
                    observed_resid = vev5_mid - intrinsic
                    prev_tv = tv.get(VEV5000Params.SYMBOL)
                    if prev_tv is None:
                        cur_tv = observed_resid
                    else:
                        alpha = 2.0 / (VEV5000Params.TV_EWMA_SPAN + 1.0)
                        cur_tv = (alpha * observed_resid
                                  + (1 - alpha) * prev_tv)
                    tv[VEV5000Params.SYMBOL] = cur_tv
                    n = tv_n.get(VEV5000Params.SYMBOL, 0) + 1
                    tv_n[VEV5000Params.SYMBOL] = n
                    if n >= VEV5000Params.MIN_SAMPLES:
                        fair = intrinsic + cur_tv
                        result[VEV5000Params.SYMBOL] = self._exec(
                            state, VEV5000Params, fair, 0)

        return result, 0, json.dumps({
            "ema": ema, "avg": avg_entry, "tv": tv, "tvn": tv_n})

    def _mr(self, state: TradingState, P, fv: float,
            ema: Dict[str, float],
            avg_entry: Dict[str, float]):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return 0, []
        target = 0
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            if prev is None:
                anchor = fv
            else:
                alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
                anchor = alpha * fv + (1 - alpha) * prev
            ema[P.SYMBOL] = anchor
            raw = -P.MR_STRENGTH * (fv - anchor)
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))

        # Profit-take overlay: scale target toward 0 when winning
        ae = avg_entry.get(P.SYMBOL, 0.0)
        decay = _profit_decay(P, pos, fv, ae)
        target = int(round(target * decay))

        return target, self._do_orders(P, fv, pos, target, od)

    def _exec(self, state: TradingState, P, fv: float, target: int) -> List[Order]:
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return []
        return self._do_orders(P, fv, pos, target, od)

    def _do_orders(self, P, fv: float, pos: int, target_pos: int,
                   od: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        skew = P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT
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
        our_bid = base_bid + shift
        our_ask = base_ask + shift
        our_bid = min(our_bid, int(fv_eff))
        our_ask = max(our_ask, int(fv_eff) + 1)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))
        return orders
