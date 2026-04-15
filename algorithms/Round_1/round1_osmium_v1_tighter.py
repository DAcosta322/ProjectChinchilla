"""Improvement 1: Tighter resting orders (9992/10008 instead of 9990/10010)."""

from datamodel import OrderDepth, TradingState, Order
from typing import List

OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = 80
ANCHOR = 10000
HALF_SPREAD = 8


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

        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < ANCHOR and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(OSMIUM, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > ANCHOR and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(OSMIUM, price, -qty))
                    sell_cap -= qty

        if buy_cap > 0:
            orders.append(Order(OSMIUM, ANCHOR - HALF_SPREAD, buy_cap))
        if sell_cap > 0:
            orders.append(Order(OSMIUM, ANCHOR + HALF_SPREAD, -sell_cap))

        return orders
