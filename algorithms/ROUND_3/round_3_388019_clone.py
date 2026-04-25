"""Round 3 SMOOTH RANGE-POSITION strategy.

Instead of bang-bang ±POS_LIMIT on local extremes (which caused
massive spread cost at each flip), target scales LINEARLY with where
the current mid sits within the rolling [lo, hi] window:

  target = POS_LIMIT * (hi + lo - 2 * fv) / (hi - lo)
          = +POS_LIMIT at fv == lo   (fully long at bottom)
          = 0           at fv == midpoint
          = -POS_LIMIT at fv == hi   (fully short at top)

Target changes a few units per tick, not 400 at a time. The existing
INV_SKEW execution crosses the book only by the small amount required
to close the gap, avoiding the 15-tick-per-unit slippage observed in
the bang-bang version.

VEVs inherit VELVET's relative target scaled to their own POS_LIMIT,
reusing the spillover idea from round_3_eff.py.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    WINDOW = 300
    MIN_RANGE = 8
    GAIN = 5                # target = GAIN * (midpoint - fv), capped


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    WINDOW = 300
    MIN_RANGE = 4
    GAIN = 10


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 3


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 3


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


def _range_target(P, fv: float, hist: List[float]) -> int:
    hist.append(fv)
    if len(hist) > P.WINDOW:
        del hist[:len(hist) - P.WINDOW]
    if len(hist) < P.WINDOW:
        return 0
    lo = min(hist)
    hi = max(hist)
    rng = hi - lo
    if rng < P.MIN_RANGE:
        return 0
    midpoint = (hi + lo) / 2.0
    raw = P.GAIN * (midpoint - fv)
    return max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))


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

        hist: Dict[str, List[float]] = td.get("hist", {})
        for P in (HydrogelParams, VelvetParams):
            hist.setdefault(P.SYMBOL, [])

        velvet_target_rel = 0.0

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                tp = _range_target(HydrogelParams, fv,
                                   hist[HydrogelParams.SYMBOL])
                result[HydrogelParams.SYMBOL] = self._orders(
                    state, HydrogelParams, fv, tp)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp = _range_target(VelvetParams, velvet_mid,
                                   hist[VelvetParams.SYMBOL])
                result[VelvetParams.SYMBOL] = self._orders(
                    state, VelvetParams, velvet_mid, tp)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    vev_target = int(round(velvet_target_rel * P.POS_LIMIT))
                    vev_target = max(-P.POS_LIMIT,
                                     min(P.POS_LIMIT, vev_target))
                    result[P.SYMBOL] = self._orders(
                        state, P, fv, vev_target)

        return result, 0, json.dumps({"hist": hist})

    def _orders(self, state: TradingState, P, fv: float,
                target_pos: int) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return orders

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