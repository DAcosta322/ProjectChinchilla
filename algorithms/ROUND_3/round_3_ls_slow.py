"""Round 3 LONG/SHORT (slow) - trend-follow with longer spans and a
signal deadband, so the book isn't crossed on noise blips.

Fixes the whipsaw problem in round_3_ls.py: FAST=500 / SLOW=3000 is
slow enough that a 100-tick wiggle doesn't flip the target, and
MIN_DIVERGENCE turns the signal off entirely when fast and slow are
close (i.e. no trend is visible yet).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    FAST_SPAN = 500
    SLOW_SPAN = 3000
    TREND_STRENGTH = 20      # target_pos per unit of (fast - slow)
    MIN_DIVERGENCE = 2.0     # |fast - slow| below this => no signal


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    FAST_SPAN = 500
    SLOW_SPAN = 3000
    TREND_STRENGTH = 50
    MIN_DIVERGENCE = 1.0


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 0
    FAST_SPAN = 0
    SLOW_SPAN = 0
    TREND_STRENGTH = 0
    MIN_DIVERGENCE = 0.0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 0
    FAST_SPAN = 0
    SLOW_SPAN = 0
    TREND_STRENGTH = 0
    MIN_DIVERGENCE = 0.0


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


class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        ema: Dict[str, float] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                ema = td.get("ema", {})
            except Exception:
                pass

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                result[HydrogelParams.SYMBOL] = self._mm_with_fair(
                    state, HydrogelParams, fv, ema)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                result[VelvetParams.SYMBOL] = self._mm_with_fair(
                    state, VelvetParams, velvet_mid, ema)

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    result[P.SYMBOL] = self._mm_with_fair(
                        state, P, fv, ema)

        return result, 0, json.dumps({"ema": ema})

    def _mm_with_fair(self, state: TradingState, P, fv: float,
                      ema: Dict[str, float]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders

        target_pos = 0
        if P.FAST_SPAN > 0 and P.SLOW_SPAN > 0:
            fast_key = P.SYMBOL + "_f"
            slow_key = P.SYMBOL + "_s"
            fprev = ema.get(fast_key)
            sprev = ema.get(slow_key)
            if fprev is None:
                fast = fv
            else:
                fa = 2.0 / (P.FAST_SPAN + 1.0)
                fast = fa * fv + (1 - fa) * fprev
            if sprev is None:
                slow = fv
            else:
                sa = 2.0 / (P.SLOW_SPAN + 1.0)
                slow = sa * fv + (1 - sa) * sprev
            ema[fast_key] = fast
            ema[slow_key] = slow

            divergence = fast - slow
            if abs(divergence) < P.MIN_DIVERGENCE:
                target_pos = 0
            else:
                raw = P.TREND_STRENGTH * divergence
                target_pos = max(-P.POS_LIMIT,
                                 min(P.POS_LIMIT, int(round(raw))))

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
