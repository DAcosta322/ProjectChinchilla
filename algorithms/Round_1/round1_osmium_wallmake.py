"""Osmium: aggressive taking + wall-tracking market making.

Takes on narrow-spread ticks (when 3rd party posts inside the bot walls).
Makes by posting just inside the bot walls (bid1+1 / ask1-1) to capture
the ~400 bot-vs-bot trades per day that happen at bid1/ask1.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = 80
MA_WINDOW = 20


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

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Compute MA fair value
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        else:
            mid = 10000
        prices.append(mid)
        if len(prices) > MA_WINDOW:
            prices[:] = prices[-MA_WINDOW:]
        fv = round(sum(prices) / len(prices))

        buy_cap = POS_LIMIT - pos
        sell_cap = POS_LIMIT + pos

        # --- Phase 1: Take mispriced orders vs MA fair value ---
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

        # --- Phase 2: Post just inside bot walls to capture flow ---
        if best_bid is not None and best_ask is not None:
            our_bid = best_bid + 1
            our_ask = best_ask - 1

            # Don't cross: if spread is already tight, fall back to walls
            if our_bid >= our_ask:
                our_bid = best_bid
                our_ask = best_ask
        else:
            our_bid = fv - 8
            our_ask = fv + 8

        if buy_cap > 0:
            orders.append(Order(OSMIUM, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(OSMIUM, our_ask, -sell_cap))

        return orders
