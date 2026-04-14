import jsonpickle

from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List

class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Generate orders for EMERALDS and TOMATOES.

        - EMERALDS: stable value; we try to stay near a short-term fair price
        - TOMATOES: fluctuating; we follow short-term momentum

        Both products have hard position limits (long + short) of 80.
        """

        # Decode the previous state (if any) so we can track price history between calls.
        history = {}
        if state.traderData:
            try:
                history = jsonpickle.decode(state.traderData)
            except Exception:
                history = {}

        last_mid = history.get("last_mid", {})

        # Constants / limits
        POSITION_LIMITS = {"EMERALDS": 80, "TOMATOES": 80}
        BASE_ORDER_SIZE = {"EMERALDS": 10, "TOMATOES": 10}

        result = {}
        new_last_mid = {}

        # Helper to cap orders to position limits (absolute).
        def cap_quantity(product: str, desired_qty: int) -> int:
            pos = state.position.get(product, 0)
            limit = POSITION_LIMITS.get(product, 0)

            # If we want to buy (desired_qty > 0), ensure absolute position stays <= limit.
            if desired_qty > 0:
                max_buy = limit - pos
                return max(0, min(desired_qty, max_buy))

            # If we want to sell (desired_qty < 0), ensure absolute position stays <= limit.
            if desired_qty < 0:
                max_sell = -limit - pos
                return min(0, max(desired_qty, max_sell))

            return 0

        for product, order_depth in state.order_depths.items():
            if product not in POSITION_LIMITS:
                # We're only trading EMERALDS and TOMATOES for this simple strategy.
                continue

            orders: List[Order] = []
            # Determine best bid/ask from the order book (first entry is best price)
            best_bid = None
            best_bid_amount = 0
            best_ask = None
            best_ask_amount = 0

            if order_depth.buy_orders:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
            if order_depth.sell_orders:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]

            # If we can't see both sides, we skip trading for this product.
            if best_bid is None or best_ask is None:
                result[product] = orders
                continue

            # Track mid price for simple signals
            mid_price = (best_bid + best_ask) / 2
            new_last_mid[product] = mid_price

            current_position = state.position.get(product, 0)

            # EMERALDS: stable, mean reversion strategy around last seen mid price.
            if product == "EMERALDS":
                reference = last_mid.get(product, mid_price)
                delta = mid_price - reference

                if delta < -0.5:
                    # price dropped: buy into strength (small size)
                    qty = cap_quantity(product, BASE_ORDER_SIZE[product])
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                elif delta > 0.5:
                    # price rose: trim position
                    qty = cap_quantity(product, -BASE_ORDER_SIZE[product])
                    if qty < 0:
                        orders.append(Order(product, best_bid, qty))

            # TOMATOES: momentum-based (follow direction of last mid price change)
            if product == "TOMATOES":
                prev_mid = last_mid.get(product, mid_price)
                momentum = mid_price - prev_mid

                if momentum > 0.1:
                    # price trending up -> buy
                    qty = cap_quantity(product, BASE_ORDER_SIZE[product])
                    if qty > 0:
                        orders.append(Order(product, best_ask, qty))
                elif momentum < -0.1:
                    # price trending down -> sell
                    qty = cap_quantity(product, -BASE_ORDER_SIZE[product])
                    if qty < 0:
                        orders.append(Order(product, best_bid, qty))

            result[product] = orders

        # Persist last mid prices for next call.
        traderData = jsonpickle.encode({"last_mid": new_last_mid})
        conversions = 0
        return result, conversions, traderData