"""Round 5 — PEBBLES using the R4 v5 chassis.

Imports the round_4_v5 trade primitives (_mr, _do_orders, _update_avg_entry,
_profit_decay, _book_top, _ofi_step, _micro_mid) and configures one PebblesXxx
params class per product. Adds basket ETF arb on top.

Why this might beat v1-v9: the R4 chassis has a robust combination that
handles drifty products in HYD/VEL — adaptive anchor (PRIOR_BLEND), trend-age
detector with hysteresis, profit-decay, INV_SKEW, MM-ladder. v1-v9 used
ad-hoc subsets that bled or whipsawed.

Tuning notes vs HYD (POS_LIMIT 200 → 10):
  - MR_STRENGTH must be ~20× smaller (signal scales with POS_LIMIT).
  - INV_SKEW kept at 3-4 (ticks, not pos-units).
  - BASE_QTY scaled down to ~3 (vs 10 in HYD).
  - FIXED_ANCHOR omitted: day 3/4 start at previous-day's close, not 10000.
    PRIOR_BLEND=0 → pure adaptive EMA.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Optional
import json
import math

# R4 v5 helpers inlined (IMC platform uploads only this single file, so
# cross-folder imports like `from round_4_v5 import ...` fail at runtime).
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


# ---- Per-product params for the 5 PEBBLES ----
class _PebblesBase:
    """Defaults tuned via sweep_pebbles_r4.py (486-config grid).

    Rank-1 BT: $129,046 across days 2/3/4
      D2 +$52,195  D3 +$44,232  D4 +$32,618 (all positive!)
      Per-product: XS +$9,631  S +$36,958  M +$24,836  L +$14,580  XL +$43,040
    """
    POS_LIMIT = 10

    # Anchor — fast 2K EMA tracks intraday drift; FIXED_ANCHOR off because
    # days 3/4 start at previous-day close, not 10000.
    ANCHOR_SPAN = 2000
    FIXED_ANCHOR = None
    PRIOR_BLEND = 0.0
    BREAK_EVEN_BLEND = 0.0

    # MR — strong (sweep peaked at 0.3). Combined with adaptive 2K anchor,
    # target hits POS_LIMIT at dev~33 ticks.
    MR_STRENGTH = 0.30
    INV_SKEW = 3.0
    BOOST_THRESHOLD = 100.0
    BOOST_GAIN = 0.3
    NEUTRAL_BAND = 20.0
    NEUTRAL_GAIN = 0.05

    # Trend-age detector — sustained drift (>200 ticks of consistent vel)
    # blends MR target toward trend-follow, weighted by maturity to FULL_AGE.
    TREND_FAST_SPAN = 200
    TREND_VEL_SPAN = 50
    TREND_THRESH = 0.005
    TREND_MIN_AGE = 200
    TREND_FULL_AGE = 2000
    TREND_STRENGTH = 0.8

    # Profit decay
    PROFIT_DIST = 30
    PROFIT_RANGE = 30

    VOL_SPAN = 200
    VOL_REF = 2.0
    OFI_GAIN = 1.0
    OFI_SPAN = 20
    DRIFT_SPAN = 1000
    DRIFT_GAIN = 0.0

    # MM — BASE_QTY=8 (sweep winner, ~80% of POS_LIMIT). Larger size posts
    # captures more of the wide PEBBLES spread per tick.
    BASE_QTY = 8
    AGG_OVERSHOOT = 2

    # MM-ladder
    MM_LADDER_MIN_SPREAD = 8
    MM_LADDER_OFFSETS = (3, 5)
    MM_LADDER_QTY = 2
    MM_LADDER_NEUTRAL_THRESH = 5

    # End-of-day / stop-loss off
    EOD_START_FRAC = 0.7
    EOD_GAIN = 0.0
    EOD_FAST_SPAN = 500
    STOP_LOSS_TICKS = 0
    DEPTH_IMB_GAIN = 0.0
    TIER2_FRAC = 0.0
    AGG_CROSS_MARGIN = 0.0


class PebblesXSParams(_PebblesBase):
    SYMBOL = "PEBBLES_XS"
    # XS+L sweep rank-1: XS keeps MR=0.3 (sweep confirms over MR=0.1
    # variant, which buys robustness at -$8K total cost). Slight TREND drop.


class PebblesSParams(_PebblesBase):
    SYMBOL = "PEBBLES_S"


class PebblesMParams(_PebblesBase):
    SYMBOL = "PEBBLES_M"


class PebblesLParams(_PebblesBase):
    """L gets a stronger MR + trend overrides — also a high-magnitude drifter
    on day 4 (L drifted -1888 from start). Sweep rank-1: MR=0.4, TREND=1.0.
    """
    SYMBOL = "PEBBLES_L"
    MR_STRENGTH = 0.40       # was 0.30
    TREND_STRENGTH = 1.0     # was 0.8


class PebblesXLParams(_PebblesBase):
    """XL gets per-product overrides — its drift magnitude (up to +4K ticks/day)
    forces stronger trend engagement than the small siblings.

    Tuned via sweep_pebbles_r4_xl.py (144-config grid, fragment-aware ranking).
    Robust-score winner: total $131,357 (+$2.3K vs shared base), worst-fragment
    -$12,826 (+$2K vs shared base), D4 +$45K (+$12K vs shared base).
    """
    SYMBOL = "PEBBLES_XL"
    TREND_STRENGTH = 1.2     # was 0.8 — engage trend follow harder on XL drifts
    TREND_MIN_AGE = 100      # was 200 — engage faster on short fragments


PEBBLES_PARAMS = [PebblesXSParams, PebblesSParams, PebblesMParams,
                  PebblesLParams, PebblesXLParams]
PEBBLES_SYMS = [P.SYMBOL for P in PEBBLES_PARAMS]


# ---- ETF basket arb config ----
class BasketArbCfg:
    BASKET_SUM = 50000
    EDGE = 0
    QTY = 10


class Trader:
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
        prev_fv = td.get("pfv", {})
        var_ewma = td.get("var", {})
        prev_book = td.get("pbook", {})
        ofi_ewma = td.get("ofi", {})
        trend = td.get("trend", {})
        fast_ema = td.get("fema", {})
        trend_age = td.get("tage", {})
        trend_sign = td.get("tsign", {})

        # Refresh avg_entry from own_trades
        for P in PEBBLES_PARAMS:
            if P.PROFIT_DIST <= 0:
                continue
            sym = P.SYMBOL
            own = state.own_trades.get(sym, []) if state.own_trades else []
            cur_pos = state.position.get(sym, 0)
            delta = sum(t.quantity if getattr(t, "buyer", None) == "SUBMISSION"
                        else -t.quantity if getattr(t, "seller", None) == "SUBMISSION" else 0
                        for t in own)
            avg_entry[sym] = _update_avg_entry(
                avg_entry.get(sym, 0.0), cur_pos - delta, own)

        # Per-product MR via R4 chassis
        for P in PEBBLES_PARAMS:
            if P.SYMBOL not in state.order_depths:
                continue
            od = state.order_depths[P.SYMBOL]
            fv = _micro_mid(od)
            if fv is None:
                continue
            _, orders = self._mr(state, P, fv, ema, avg_entry, prev_fv,
                                 var_ewma, prev_book, ofi_ewma, trend,
                                 fast_ema, trend_age, trend_sign)
            result[P.SYMBOL] = orders

        # ---- ETF basket arb on top ----
        # Compute leftover capacity AFTER per-product orders staged.
        books = {}
        valid_basket = True
        for sym in PEBBLES_SYMS:
            od = state.order_depths.get(sym)
            if not od or not od.buy_orders or not od.sell_orders:
                valid_basket = False
                break
            books[sym] = (max(od.buy_orders.keys()),
                          min(od.sell_orders.keys()),
                          od.buy_orders[max(od.buy_orders.keys())],
                          -od.sell_orders[min(od.sell_orders.keys())])

        if valid_basket:
            sum_bb = sum(books[s][0] for s in PEBBLES_SYMS)
            sum_ba = sum(books[s][1] for s in PEBBLES_SYMS)

            if sum_bb > BasketArbCfg.BASKET_SUM + BasketArbCfg.EDGE:
                # Lock-sell basket. Cap qty by per-product remaining sell capacity
                # accounting for orders already staged.
                qty = BasketArbCfg.QTY
                for sym in PEBBLES_SYMS:
                    bv = books[sym][2]
                    pos = state.position.get(sym, 0)
                    staged = sum(-o.quantity for o in result.get(sym, [])
                                 if o.quantity < 0)
                    sell_cap = _PebblesBase.POS_LIMIT + pos - staged
                    qty = min(qty, bv, max(0, sell_cap))
                if qty > 0:
                    for sym in PEBBLES_SYMS:
                        bb = books[sym][0]
                        result.setdefault(sym, []).append(Order(sym, bb, -qty))

            if sum_ba < BasketArbCfg.BASKET_SUM - BasketArbCfg.EDGE:
                qty = BasketArbCfg.QTY
                for sym in PEBBLES_SYMS:
                    av = books[sym][3]
                    pos = state.position.get(sym, 0)
                    staged = sum(o.quantity for o in result.get(sym, [])
                                 if o.quantity > 0)
                    buy_cap = _PebblesBase.POS_LIMIT - pos - staged
                    qty = min(qty, av, max(0, buy_cap))
                if qty > 0:
                    for sym in PEBBLES_SYMS:
                        ba = books[sym][1]
                        result.setdefault(sym, []).append(Order(sym, ba, qty))

        return result, 0, json.dumps({
            "ema": ema, "avg": avg_entry,
            "pfv": prev_fv, "var": var_ewma,
            "pbook": prev_book, "ofi": ofi_ewma,
            "trend": trend, "fema": fast_ema,
            "tage": trend_age, "tsign": trend_sign,
        })

    # ---- Copied near-verbatim from round_4_v5._mr (and stripped of bot-event
    # / voucher branches). Preserves the chassis intact. ----
    def _mr(self, state, P, fv, ema, avg_entry, prev_fv, var_ewma,
            prev_book, ofi_ewma, trend, fast_ema, trend_age, trend_sign):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return 0, []

        vol_factor = 1.0
        if P.VOL_SPAN > 0:
            if P.SYMBOL in prev_fv:
                ret = fv - prev_fv[P.SYMBOL]
                ret = max(-50.0, min(50.0, ret))
                prev_var = var_ewma.get(P.SYMBOL, P.VOL_REF * P.VOL_REF)
                a = 2.0 / (P.VOL_SPAN + 1.0)
                var_ewma[P.SYMBOL] = a * (ret * ret) + (1 - a) * prev_var
            prev_fv[P.SYMBOL] = fv
            std_est = math.sqrt(max(0.01, var_ewma.get(P.SYMBOL, 1.0)))
            vol_factor = max(0.5, min(2.0, P.VOL_REF / std_est))

        ofi_smooth = 0.0
        if getattr(P, 'OFI_GAIN', 0) > 0 and getattr(P, 'OFI_SPAN', 0) > 0:
            cur_book = _book_top(od)
            ofi_one = _ofi_step(prev_book.get(P.SYMBOL), cur_book)
            prev_book[P.SYMBOL] = cur_book
            a_ofi = 2.0 / (P.OFI_SPAN + 1.0)
            ofi_smooth = a_ofi * ofi_one + (1 - a_ofi) * ofi_ewma.get(P.SYMBOL, 0.0)
            ofi_ewma[P.SYMBOL] = ofi_smooth

        depth_imb = 0.0
        if getattr(P, 'DEPTH_IMB_GAIN', 0) > 0:
            sum_bid = sum(od.buy_orders.values())
            sum_ask = -sum(od.sell_orders.values())
            tot = sum_bid + sum_ask
            if tot > 0:
                depth_imb = (sum_bid - sum_ask) / tot

        drift_signal = 0.0
        drift_gain = getattr(P, "DRIFT_GAIN", 0.0)
        if drift_gain != 0:
            drift_key = P.SYMBOL + "_DRIFT"
            drift_span = getattr(P, "DRIFT_SPAN", 1000)
            prev_drift = fast_ema.get(drift_key)
            a_drift = 2.0 / (drift_span + 1.0)
            cur_drift_ema = fv if prev_drift is None else (
                a_drift * fv + (1 - a_drift) * prev_drift)
            fast_ema[drift_key] = cur_drift_ema
            drift_signal = fv - cur_drift_ema

        target = 0
        if P.ANCHOR_SPAN > 0:
            prev_L = ema.get(P.SYMBOL)
            alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
            if prev_L is None:
                fa = getattr(P, "FIXED_ANCHOR", None)
                L = float(fa) if fa is not None else fv
            else:
                L = alpha * fv + (1 - alpha) * prev_L
            ema[P.SYMBOL] = L
            anchor = L
            prior_blend = getattr(P, "PRIOR_BLEND", 0.0)
            fa_val = getattr(P, "FIXED_ANCHOR", None)
            if prior_blend > 0 and fa_val is not None:
                anchor = (1 - prior_blend) * anchor + prior_blend * float(fa_val)

            be_blend = getattr(P, "BREAK_EVEN_BLEND", 0.0)
            if be_blend > 0:
                ae = avg_entry.get(P.SYMBOL, 0.0)
                if pos != 0 and ae > 0:
                    pos_frac = min(1.0, abs(pos) / P.POS_LIMIT)
                    weight_be = be_blend * pos_frac
                    anchor = (1 - weight_be) * anchor + weight_be * ae

            mr_eff = P.MR_STRENGTH * vol_factor
            dev = fv - anchor
            bt_th = getattr(P, "BOOST_THRESHOLD", 1e9)
            bg = getattr(P, "BOOST_GAIN", 0.0)
            boost = 0.0
            if bg > 0 and abs(dev) > bt_th:
                excess = abs(dev) - bt_th
                boost = -bg * (1.0 if dev > 0 else -1.0) * excess
            n_band = getattr(P, "NEUTRAL_BAND", 0.0)
            n_gain = getattr(P, "NEUTRAL_GAIN", 0.0)
            neutral_pull = 0.0
            if n_band > 0 and n_gain > 0 and abs(dev) < n_band:
                neutral_pull = -n_gain * pos * (1.0 - abs(dev) / n_band)
            raw = (-mr_eff * dev + boost + neutral_pull
                   + getattr(P, 'OFI_GAIN', 0) * ofi_smooth
                   + getattr(P, 'DEPTH_IMB_GAIN', 0) * depth_imb
                   + drift_gain * drift_signal)
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))

            # Trend-age modulation
            tr_thresh = getattr(P, "TREND_THRESH", 0.0)
            tr_min_age = getattr(P, "TREND_MIN_AGE", 0)
            tr_full_age = getattr(P, "TREND_FULL_AGE", 1)
            tr_strength = getattr(P, "TREND_STRENGTH", 0.0)
            tr_vel_span = getattr(P, "TREND_VEL_SPAN", 50)
            tr_fast_span = getattr(P, "TREND_FAST_SPAN", 0)
            if tr_thresh > 0 and tr_strength > 0 and tr_fast_span > 0:
                key = P.SYMBOL + "_TFAST"
                prev_tf = fast_ema.get(key)
                a_tf = 2.0 / (tr_fast_span + 1.0)
                if prev_tf is None:
                    fa = getattr(P, "FIXED_ANCHOR", None)
                    prev_tf = float(fa) if fa is not None else fv
                Tf = a_tf * fv + (1 - a_tf) * prev_tf
                fast_ema[key] = Tf
                velocity_raw = Tf - prev_tf
                prev_vel = trend.get(P.SYMBOL + "_VEL", 0.0)
                bv = 2.0 / (tr_vel_span + 1.0)
                velocity = bv * velocity_raw + (1.0 - bv) * prev_vel
                trend[P.SYMBOL + "_VEL"] = velocity

                if velocity > tr_thresh:
                    cur_sign = 1
                elif velocity < -tr_thresh:
                    cur_sign = -1
                else:
                    cur_sign = 0
                prev_age = trend_age.get(P.SYMBOL, 0)
                prev_sign = trend_sign.get(P.SYMBOL, 0)
                if cur_sign == 0:
                    new_age = max(0, prev_age - 2)
                    new_sign = prev_sign if new_age > 0 else 0
                elif cur_sign == prev_sign:
                    new_age = prev_age + 1
                    new_sign = cur_sign
                else:
                    new_age = 1
                    new_sign = cur_sign
                trend_age[P.SYMBOL] = new_age
                trend_sign[P.SYMBOL] = new_sign

                if new_age > tr_min_age and new_sign != 0:
                    span_a = max(1, tr_full_age - tr_min_age)
                    maturity = max(0.0, min(1.0, (new_age - tr_min_age) / span_a))
                    follow_w = tr_strength * maturity
                    follow_target = new_sign * P.POS_LIMIT
                    target = int(round((1 - follow_w) * target + follow_w * follow_target))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))

        decay = _profit_decay(P, pos, fv, avg_entry.get(P.SYMBOL, 0.0))
        target = int(round(target * decay))

        return target, self._do_orders(P, fv, pos, target, od)

    def _do_orders(self, P, fv, pos, target_pos, od):
        orders: List[Order] = []
        skew = P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        agg_over = getattr(P, "AGG_OVERSHOOT", 50)
        agg_buy_cap = max(0, min(buy_cap, (target_pos - pos) + agg_over))
        agg_sell_cap = max(0, min(sell_cap, (pos - target_pos) + agg_over))

        agg_margin = getattr(P, "AGG_CROSS_MARGIN", 0.0)
        for price in sorted(od.sell_orders.keys()):
            if price < fv_eff - agg_margin and agg_buy_cap > 0:
                qty = min(-od.sell_orders[price], agg_buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                agg_buy_cap -= qty
                buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fv_eff + agg_margin and agg_sell_cap > 0:
                qty = min(od.buy_orders[price], agg_sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty))
                agg_sell_cap -= qty
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

        base = getattr(P, "BASE_QTY", 10)
        eff_base = max(1, base)
        bid_qty = min(buy_cap, max(0, target_pos - pos) + eff_base)
        ask_qty = min(sell_cap, max(0, pos - target_pos) + eff_base)

        if bid_qty > 0:
            orders.append(Order(P.SYMBOL, our_bid, bid_qty))
        if ask_qty > 0:
            orders.append(Order(P.SYMBOL, our_ask, -ask_qty))

        ladder_min = getattr(P, "MM_LADDER_MIN_SPREAD", 0)
        ladder_offsets = getattr(P, "MM_LADDER_OFFSETS", ())
        ladder_qty = getattr(P, "MM_LADDER_QTY", 0)
        ladder_thresh = getattr(P, "MM_LADDER_NEUTRAL_THRESH", 0)
        spread_now = best_ask - best_bid
        if (ladder_min > 0 and spread_now > ladder_min and ladder_qty > 0
                and abs(target_pos) <= ladder_thresh):
            buy_cap_left = buy_cap - bid_qty
            sell_cap_left = sell_cap - ask_qty
            for off in ladder_offsets:
                lbid = best_bid + off
                lask = best_ask - off
                if lbid < lask:
                    if buy_cap_left > 0 and lbid > our_bid:
                        q = min(buy_cap_left, ladder_qty)
                        if q > 0:
                            orders.append(Order(P.SYMBOL, lbid, q))
                            buy_cap_left -= q
                    if sell_cap_left > 0 and lask < our_ask:
                        q = min(sell_cap_left, ladder_qty)
                        if q > 0:
                            orders.append(Order(P.SYMBOL, lask, -q))
                            sell_cap_left -= q
        return orders
