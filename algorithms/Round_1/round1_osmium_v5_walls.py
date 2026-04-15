"""Improvement 5: Dynamic wall-tracking (bid1+1 / ask1-1) to capture bot flow."""

from datamodel import OrderDepth, TradingState, Order
from typing import List

OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = 80
ANCHOR = 10000


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

        # Post just inside the bot walls (bid1+1 / ask1-1)
        if od.buy_orders and od.sell_orders:
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            our_bid = best_bid + 1
            our_ask = best_ask - 1

            # Don't cross ourselves
            if our_bid >= our_ask:
                our_bid = best_bid
                our_ask = best_ask
        else:
            our_bid = ANCHOR - 8
            our_ask = ANCHOR + 8

        if buy_cap > 0:
            orders.append(Order(OSMIUM, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(OSMIUM, our_ask, -sell_cap))

        return orders
