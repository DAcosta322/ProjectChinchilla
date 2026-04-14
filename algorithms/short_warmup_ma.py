"""Short warm-up dual-MA strategy.

Copy of tutorial.py with the TOMATOES StockTrader modified:
- SLOW_WINDOW reduced from 500 to 200 (signal activates after ~10 ticks)
- SIGNAL_MULT = 1.0 so FV = fast_ma (no lag above market in downtrends)
- Inventory penalty (GAMMA) skews FV against position to prevent drift
- Hard clearing at CLEAR_THRESHOLD to avoid hitting position limits
- All EMERALDS logic identical to tutorial.py

Key behavioral insight (from 97620 analysis):
  SIGNAL_MULT < 1 causes FV to lag above mid in downtrends, creating a
  persistent buy bias. SIGNAL_MULT = 1.0 makes FV = fast_ma which tracks
  the market neutrally. The inventory penalty handles the rest.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

COMMODITY_SYMBOL = "EMERALDS"
STOCK_SYMBOL = "TOMATOES"

POS_LIMITS = {
    COMMODITY_SYMBOL: 80,
    STOCK_SYMBOL: 80,
}


class ProductTrader:
    def __init__(self, name, fair_value, pos_limit):
        self.name = name
        self.fair_value = fair_value
        self.pos_limit = pos_limit

    def get_position(self, state: TradingState) -> int:
        return state.position.get(self.name, 0)

    def get_order_depth(self, state: TradingState) -> OrderDepth:
        return state.order_depths.get(self.name, OrderDepth())

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            return (best_bid + best_ask) / 2
        return self.fair_value

    def run(self, state: TradingState) -> List[Order]:
        raise NotImplementedError


class CommodityTrader(ProductTrader):
    """Identical to tutorial.py CommodityTrader."""

    def __init__(self):
        super().__init__(COMMODITY_SYMBOL, fair_value=10000, pos_limit=POS_LIMITS[COMMODITY_SYMBOL])

    def run(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < self.fair_value and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > self.fair_value and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

        EMERALDS_DIFF = 7
        if buy_capacity > 0:
            orders.append(Order(self.name, self.fair_value - EMERALDS_DIFF, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, self.fair_value + EMERALDS_DIFF, -sell_capacity))

        return orders


class StockTrader(ProductTrader):
    """Short warm-up dual-MA strategy for TOMATOES.

    Key fixes vs tutorial (informed by 54133 + 97620 analysis):
    1. SLOW_WINDOW 500->200: signal activates after ~10 ticks, not ~500
    2. SIGNAL_MULT = 1.0: FV = fast_ma, no lag above market in downtrends
       (SIGNAL_MULT < 1 caused 97620's persistent long bias — FV sat above
        mid in every downtrend, making the algo buy into drops)
    3. Inventory penalty (GAMMA): skews FV against position to prevent drift
    4. Lower CLEAR_THRESHOLD: starts unwinding earlier
    """

    # --- Tunable parameters ---
    FAST_WINDOW = 10     # Smooth enough to filter tick noise, fast enough to catch trends
    SLOW_WINDOW = 200    # Activates after ~10 ticks; 200-tick baseline is stable
    SIGNAL_MULT = 1.0    # FV = fast_ma. No mean-reversion bias (the 97620 bug)
    SPREAD = 3           # Wide resting orders = less adverse selection
    CLEAR_THRESHOLD = 40 # Start unwinding early to avoid getting trapped at ±80
    GAMMA = 0.05         # Inventory penalty: adjusts FV by -GAMMA*position

    def __init__(self):
        super().__init__(STOCK_SYMBOL, fair_value=5000, pos_limit=POS_LIMITS[STOCK_SYMBOL])

    def compute_fair_value(self, order_depth: OrderDepth, price_history: List[float]) -> float:
        mid = self.get_mid_price(order_depth)
        price_history.append(mid)

        if len(price_history) > self.SLOW_WINDOW:
            price_history[:] = price_history[-self.SLOW_WINDOW:]

        # Use signal as soon as we have more data than FAST_WINDOW
        if len(price_history) > self.FAST_WINDOW:
            fast_ma = sum(price_history[-self.FAST_WINDOW:]) / self.FAST_WINDOW
            slow_ma = sum(price_history) / len(price_history)
            signal = fast_ma - slow_ma
            return slow_ma + signal * self.SIGNAL_MULT
        return mid

    def run(self, state: TradingState, price_history: List[float]) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        raw_fv = self.compute_fair_value(order_depth, price_history)

        # Inventory penalty: push FV against our position
        # Long -> FV drops -> more sells, fewer buys
        # Short -> FV rises -> more buys, fewer sells
        adjusted_fv = raw_fv - self.GAMMA * position
        fv = round(adjusted_fv)

        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        # Take cheap sells
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fv and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        # Take expensive buys
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fv and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

        # Inventory clearing when near limit
        filled_buy = self.pos_limit - position - buy_capacity
        filled_sell = self.pos_limit + position - sell_capacity
        current_pos = position + filled_buy - filled_sell

        if current_pos > self.CLEAR_THRESHOLD:
            clear_qty = min(current_pos - self.CLEAR_THRESHOLD, sell_capacity)
            if clear_qty > 0:
                orders.append(Order(self.name, fv, -clear_qty))
                sell_capacity -= clear_qty
        elif current_pos < -self.CLEAR_THRESHOLD:
            clear_qty = min(-current_pos - self.CLEAR_THRESHOLD, buy_capacity)
            if clear_qty > 0:
                orders.append(Order(self.name, fv, clear_qty))
                buy_capacity -= clear_qty

        # Post resting orders
        if buy_capacity > 0:
            orders.append(Order(self.name, fv - self.SPREAD, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, fv + self.SPREAD, -sell_capacity))

        return orders


class Trader:

    def __init__(self):
        self.emerald_trader = CommodityTrader()
        self.tomato_trader = StockTrader()

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result = {}

        tomato_prices: List[float] = []
        if state.traderData:
            try:
                tomato_prices = jsonpickle.decode(state.traderData)
            except Exception:
                tomato_prices = []

        if COMMODITY_SYMBOL in state.order_depths:
            result[COMMODITY_SYMBOL] = self.emerald_trader.run(state)

        if STOCK_SYMBOL in state.order_depths:
            result[STOCK_SYMBOL] = self.tomato_trader.run(state, tomato_prices)

        traderData = jsonpickle.encode(tomato_prices)
        conversions = 0
        return result, conversions, traderData
