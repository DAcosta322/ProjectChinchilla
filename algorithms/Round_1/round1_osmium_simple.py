"""ASH_COATED_OSMIUM simple market maker.

Osmium is fixed at 10000 with spread 20.
Buy everything below 10000, sell everything above 10000.
Post resting orders at the band edges (9990 bid, 10010 ask).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = 80
ANCHOR = 10000
HALF_SPREAD = 10


class Trader:

    def run(self, state: TradingState):
        result = {}
        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade(state)
        return result, 0, ""

    def _trade(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[OSMIUM]
        pos = state.position.get(OSMIUM, 0)
        buy_cap = POS_LIMIT - pos
        sell_cap = POS_LIMIT + pos

        # Take any asks below anchor
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < ANCHOR and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(OSMIUM, price, qty))
                    buy_cap -= qty

        # Take any bids above anchor
        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > ANCHOR and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(OSMIUM, price, -qty))
                    sell_cap -= qty

        # Rest at band edges
        if buy_cap > 0:
            orders.append(Order(OSMIUM, ANCHOR - HALF_SPREAD, buy_cap))
        if sell_cap > 0:
            orders.append(Order(OSMIUM, ANCHOR + HALF_SPREAD, -sell_cap))

        return orders
