"""Round 3 MR + VOLATILITY-SCALED STOP.

Extension of round_3_sl.py: stop distance is not a fixed number of
ticks; it is STOP_SIGMA * rolling_std, where rolling_std is an EWMA
estimate of |fv - anchor|. On a low-volatility day the stop tightens
automatically; on a high-volatility day it loosens. The point is to
keep the stop at the same "statistical significance" regardless of
the day, rather than picking a fixed constant that is too tight on
d1 (std 38) but too loose on d3 (std 30).

Rolling variance:
  var_t = alpha * (fv - anchor)^2 + (1 - alpha) * var_{t-1}
  alpha = 2 / (VOL_SPAN + 1)

Stop fires when adverse move > STOP_SIGMA * sqrt(var_t).

Re-entry gate + cooldown same as round_3_sl.py.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json
import math


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    ANCHOR_SPAN = 500
    MR_STRENGTH = 5
    VOL_SPAN = 500
    STOP_SIGMA = 2.0
    STOP_COOLDOWN = 200


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    VOL_SPAN = 0            # feature off: VELVET MR is already clean
    STOP_SIGMA = 0
    STOP_COOLDOWN = 0


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 0
    ANCHOR_SPAN = 0
    MR_STRENGTH = 0
    VOL_SPAN = 0
    STOP_SIGMA = 0
    STOP_COOLDOWN = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 0
    ANCHOR_SPAN = 0
    MR_STRENGTH = 0
    VOL_SPAN = 0
    STOP_SIGMA = 0
    STOP_COOLDOWN = 0


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
    return avg, cur_pos


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
        var: Dict[str, float] = td.get("var", {})
        avg_entry: Dict[str, float] = td.get("avg", {})
        stop_cool: Dict[str, int] = td.get("cool", {})
        stop_px: Dict[str, float] = td.get("sp", {})
        stop_side: Dict[str, int] = td.get("ss", {})

        for P in (HydrogelParams, VelvetParams, VEV4000Params, VEV4500Params):
            if P.STOP_SIGMA <= 0:
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
            new_avg, _ = _update_avg_entry(prev_avg, pre_pos, own)
            avg_entry[sym] = new_avg

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                result[HydrogelParams.SYMBOL] = self._trade(
                    state, HydrogelParams, fv, ema, var, avg_entry,
                    stop_cool, stop_px, stop_side)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                result[VelvetParams.SYMBOL] = self._trade(
                    state, VelvetParams, velvet_mid, ema, var, avg_entry,
                    stop_cool, stop_px, stop_side)

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    result[P.SYMBOL] = self._trade(
                        state, P, fv, ema, var, avg_entry,
                        stop_cool, stop_px, stop_side)

        out = {"ema": ema, "var": var, "avg": avg_entry,
               "cool": stop_cool, "sp": stop_px, "ss": stop_side}
        return result, 0, json.dumps(out)

    def _trade(self, state: TradingState, P, fv: float,
               ema: Dict[str, float], var: Dict[str, float],
               avg_entry: Dict[str, float],
               stop_cool: Dict[str, int],
               stop_px: Dict[str, float],
               stop_side: Dict[str, int]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders

        target_pos = 0
        anchor = fv
        if P.ANCHOR_SPAN > 0:
            prev = ema.get(P.SYMBOL)
            if prev is None:
                anchor = fv
            else:
                alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
                anchor = alpha * fv + (1 - alpha) * prev
            ema[P.SYMBOL] = anchor
            raw = -P.MR_STRENGTH * (fv - anchor)
            target_pos = max(-P.POS_LIMIT,
                             min(P.POS_LIMIT, int(round(raw))))

        # Rolling variance estimate (EWMA of squared residual vs anchor)
        std = 0.0
        if P.VOL_SPAN > 0 and P.STOP_SIGMA > 0:
            vprev = var.get(P.SYMBOL)
            resid_sq = (fv - anchor) ** 2
            if vprev is None:
                v = resid_sq
            else:
                va = 2.0 / (P.VOL_SPAN + 1.0)
                v = va * resid_sq + (1 - va) * vprev
            var[P.SYMBOL] = v
            std = math.sqrt(v)

        # Volatility-scaled stop
        if P.STOP_SIGMA > 0 and pos != 0 and std > 0:
            ae = avg_entry.get(P.SYMBOL, 0.0)
            if ae > 0:
                loss_per_unit = (ae - fv) if pos > 0 else (fv - ae)
                if loss_per_unit > P.STOP_SIGMA * std:
                    target_pos = 0
                    stop_cool[P.SYMBOL] = P.STOP_COOLDOWN
                    stop_px[P.SYMBOL] = fv
                    stop_side[P.SYMBOL] = 1 if pos > 0 else -1

        # Cooldown: prevent re-entry on stopped-out side until price
        # recovers through stop_px
        cool = stop_cool.get(P.SYMBOL, 0)
        if cool > 0:
            side = stop_side.get(P.SYMBOL, 0)
            sp = stop_px.get(P.SYMBOL, 0.0)
            gate_clear = (side > 0 and fv > sp) or (side < 0 and fv < sp)
            if not gate_clear:
                if side > 0:
                    target_pos = min(target_pos, 0)
                elif side < 0:
                    target_pos = max(target_pos, 0)
            stop_cool[P.SYMBOL] = cool - 1

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
