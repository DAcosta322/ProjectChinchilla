"""Round 3 MR + QUADRATIC inventory skew.

Base: round_3.py MR framework.

Change: fv_eff skew is split into a linear term (unchanged from
baseline) plus a quadratic term that ramps up at extreme inventories.

  rel  = (pos - target_pos) / POS_LIMIT        in [-1, +1]
  skew = INV_SKEW_LIN  * rel
       + INV_SKEW_QUAD * rel * |rel|

At rel near 0 (inventory close to target) the quadratic contribution
is tiny so normal MM spread applies and we collect edge. At rel near
+/-1 (inventory at extremes) the quadratic dominates and fv_eff is
shifted enough to force the book to unwind us - like a volatility-
independent panic button that only kicks in once we are deep in
adverse inventory.

Motivation: the prior feedback 'algo piles on one-sided' is an
inventory-management problem. Linear skew at INV_SKEW=15 is already
enough at +/- POS_LIMIT but gives too much slack at intermediate
inventories. Quadratic skew keeps intermediate slack small and
extreme pressure large.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional
import json


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW_LIN = 15       # same as baseline
    INV_SKEW_QUAD = 5       # small extra only at very high |rel|
    ANCHOR_SPAN = 500
    MR_STRENGTH = 5


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW_LIN = 3
    INV_SKEW_QUAD = 1
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW_LIN = 0
    INV_SKEW_QUAD = 0
    ANCHOR_SPAN = 0
    MR_STRENGTH = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW_LIN = 0
    INV_SKEW_QUAD = 0
    ANCHOR_SPAN = 0
    MR_STRENGTH = 0


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

        # Linear + quadratic inventory skew
        rel = (pos - target_pos) / P.POS_LIMIT
        skew = P.INV_SKEW_LIN * rel + P.INV_SKEW_QUAD * rel * abs(rel)
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
