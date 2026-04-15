"""ASH_COATED_OSMIUM heatmap strategy.

Osmium is anchored at 10000 with a spread of ~20 (range 9990–10010).
Use the short-term MA position within that band to set a target position:
  - MA near 9990 → target +80 (max long, price is cheap)
  - MA near 10010 → target -80 (max short, price is expensive)
Always take beneficial fills (buy below fair value, sell above).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

OSMIUM = "ASH_COATED_OSMIUM"
POS_LIMIT = 80

# Band parameters
ANCHOR = 10000
HALF_SPREAD = 35  # band is ANCHOR ± HALF_SPREAD

# MA window for estimating current price level
MA_WINDOW = 80


class Trader:

    def __init__(self):
        pass

    def run(self, state: TradingState):
        result = {}

        trader_state: dict = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        prices: List[float] = trader_state.get("prices", [])

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, prices)

        trader_state["prices"] = prices

        return result, 0, json.dumps(trader_state)

    def _trade_osmium(self, state: TradingState, prices: List[float]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[OSMIUM]
        position = state.position.get(OSMIUM, 0)

        # Compute mid price and update MA history
        mid = self._mid(od)
        prices.append(mid)
        if len(prices) > MA_WINDOW:
            prices[:] = prices[-MA_WINDOW:]

        ma = sum(prices) / len(prices)

        # --- Heatmap: map MA position in band to a target position ---
        # t=0 at bottom of band (9990) → target +80
        # t=1 at top of band (10010) → target -80
        # Clamp to [0,1] so we stay within limits even if MA drifts outside band
        t = (ma - (ANCHOR - HALF_SPREAD)) / (2 * HALF_SPREAD)
        t = max(0.0, min(1.0, t))
        target_pos = round(POS_LIMIT - 2 * POS_LIMIT * t)  # +80 to -80

        buy_capacity = POS_LIMIT - position
        sell_capacity = POS_LIMIT + position

        # --- Phase 1: Take any beneficial orders from the book ---
        # Buy anything offered below the anchor
        if od.sell_orders:
            for ask_price in sorted(od.sell_orders.keys()):
                if ask_price < ANCHOR and buy_capacity > 0:
                    vol = -od.sell_orders[ask_price]
                    qty = min(vol, buy_capacity)
                    orders.append(Order(OSMIUM, ask_price, qty))
                    buy_capacity -= qty
                    position += qty

        # Sell into any bid above the anchor
        if od.buy_orders:
            for bid_price in sorted(od.buy_orders.keys(), reverse=True):
                if bid_price > ANCHOR and sell_capacity > 0:
                    vol = od.buy_orders[bid_price]
                    qty = min(vol, sell_capacity)
                    orders.append(Order(OSMIUM, bid_price, -qty))
                    sell_capacity -= qty
                    position -= qty

        # --- Phase 2: Place resting orders to move toward the target position ---
        delta = target_pos - position

        if delta > 0 and buy_capacity > 0:
            # We want to buy more — bid aggressively based on how far from target
            qty = min(delta, buy_capacity)
            # Bid closer to anchor when we're far from target, back off when close
            aggression = min(1.0, abs(delta) / POS_LIMIT)
            bid_price = round(ANCHOR - HALF_SPREAD + aggression * (HALF_SPREAD - 1))
            orders.append(Order(OSMIUM, bid_price, qty))
            buy_capacity -= qty

        elif delta < 0 and sell_capacity > 0:
            qty = min(-delta, sell_capacity)
            aggression = min(1.0, abs(delta) / POS_LIMIT)
            ask_price = round(ANCHOR + HALF_SPREAD - aggression * (HALF_SPREAD - 1))
            orders.append(Order(OSMIUM, ask_price, -qty))
            sell_capacity -= qty

        # --- Phase 3: Use remaining capacity for passive market-making ---
        if buy_capacity > 0:
            orders.append(Order(OSMIUM, ANCHOR - HALF_SPREAD, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(OSMIUM, ANCHOR + HALF_SPREAD, -sell_capacity))

        return orders

    @staticmethod
    def _mid(od: OrderDepth) -> float:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2
        return ANCHOR
