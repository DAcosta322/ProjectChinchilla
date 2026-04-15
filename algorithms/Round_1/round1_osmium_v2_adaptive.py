"""Improvement 2: Adaptive FV via rolling MA instead of fixed 10000."""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = 80
MA_WINDOW = 20
HALF_SPREAD = 8


class Trader:

    def run(self, state: TradingState):
        result = {}
        prices = []
        if state.traderData:
            try:
                prices = json.loads(state.traderData)
            except Exception:
                prices = []

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade(state, prices)

        return result, 0, json.dumps(prices)

    def _trade(self, state: TradingState, prices: List[float]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[OSMIUM]
        pos = state.position.get(OSMIUM, 0)

        # Compute mid
        if od.buy_orders and od.sell_orders:
            mid = (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
        else:
            mid = 10000

        prices.append(mid)
        if len(prices) > MA_WINDOW:
            prices[:] = prices[-MA_WINDOW:]

        fv = round(sum(prices) / len(prices))

        buy_cap = POS_LIMIT - pos
        sell_cap = POS_LIMIT + pos

        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fv and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(OSMIUM, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fv and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(OSMIUM, price, -qty))
                    sell_cap -= qty

        if buy_cap > 0:
            orders.append(Order(OSMIUM, fv - HALF_SPREAD, buy_cap))
        if sell_cap > 0:
            orders.append(Order(OSMIUM, fv + HALF_SPREAD, -sell_cap))

        return orders
