from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle

### General ### General ### General ### General ### General ###

COMMODITY_SYMBOL = "EMERALDS"
STOCK_SYMBOL = "TOMATOES"

POS_LIMITS = {
    COMMODITY_SYMBOL: 80,
    STOCK_SYMBOL: 80
}

### General ### General ### General ### General ### General ###
### Utilities ### Utilities ### Utilities ### Utilities ### Utilities ###

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


### Utilities ### Utilities ### Utilities ### Utilities ### Utilities ###
### Commodity ### Commodity ### Commodity ### Commodity ### Commodity ###

class CommodityTrader(ProductTrader):
    """Trades EMERALDS using market-making around the stable fair value of 10,000.

    Strategy: EMERALDS are stable, so we market-make around 10,000.
    - Take any sell orders priced below fair value (cheap buys).
    - Take any buy orders priced above fair value (expensive sells).
    - Post resting orders at fair_value-1 (bid) and fair_value+1 (ask)
      to capture the spread on remaining capacity.
    """

    def __init__(self):
        super().__init__(COMMODITY_SYMBOL, fair_value=10000, pos_limit=POS_LIMITS[COMMODITY_SYMBOL])

    def run(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        # Take cheap sell orders (buy below fair value)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < self.fair_value and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]  # sell volumes are negative
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        # Take expensive buy orders (sell above fair value)
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > self.fair_value and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

        # Post resting orders to capture spread on remaining capacity
        if buy_capacity > 0:
            orders.append(Order(self.name, self.fair_value - 1, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, self.fair_value + 1, -sell_capacity))

        return orders


### Commodity ### Commodity ### Commodity ### Commodity ### Commodity ###
### Stock ### Stock ### Stock ### Stock ### Stock ###

class StockTrader(ProductTrader):
    """Trades TOMATOES using dual-MA momentum-adjusted fair value.

    Strategy: Use a fast MA and slow MA to detect price momentum.
    The fair value is the slow MA shifted by the momentum signal
    (fast_ma - slow_ma) scaled by a multiplier. This allows the
    strategy to follow trends rather than fight them, while still
    capturing mean-reversion during range-bound periods.
    """

    FAST_WINDOW = 8
    SLOW_WINDOW = 500
    SIGNAL_MULT = 0.5

    def __init__(self):
        super().__init__(STOCK_SYMBOL, fair_value=5000, pos_limit=POS_LIMITS[STOCK_SYMBOL])

    def compute_fair_value(self, order_depth: OrderDepth, price_history: List[float]) -> float:
        mid = self.get_mid_price(order_depth)
        price_history.append(mid)

        # Keep only the last SLOW_WINDOW entries
        if len(price_history) > self.SLOW_WINDOW:
            price_history[:] = price_history[-self.SLOW_WINDOW:]

        if len(price_history) >= self.SLOW_WINDOW:
            fast_ma = sum(price_history[-self.FAST_WINDOW:]) / self.FAST_WINDOW
            slow_ma = sum(price_history) / len(price_history)
            signal = fast_ma - slow_ma
            return slow_ma + signal * self.SIGNAL_MULT
        else:
            return mid

    def run(self, state: TradingState, price_history: List[float]) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        fair_value = self.compute_fair_value(order_depth, price_history)
        fair_value_rounded = round(fair_value)

        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        # Take cheap sell orders (buy below fair value)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair_value_rounded and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        # Take expensive buy orders (sell above fair value)
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair_value_rounded and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

        # Post resting orders around fair value
        if buy_capacity > 0:
            orders.append(Order(self.name, fair_value_rounded - 2, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, fair_value_rounded + 2, -sell_capacity))

        return orders


### Stock ### Stock ### Stock ### Stock ### Stock ###
### Trader ### Trader ### Trader ### Trader ### Trader ###

class Trader:

    def __init__(self):
        self.emerald_trader = CommodityTrader()
        self.tomato_trader = StockTrader()

    def bid(self):
        return 15

    def run(self, state: TradingState):
        result = {}

        # Restore persisted state (tomato price history)
        tomato_prices: List[float] = []
        if state.traderData:
            try:
                tomato_prices = jsonpickle.decode(state.traderData)
            except Exception:
                tomato_prices = []

        # Run each product's strategy
        if COMMODITY_SYMBOL in state.order_depths:
            result[COMMODITY_SYMBOL] = self.emerald_trader.run(state)

        if STOCK_SYMBOL in state.order_depths:
            result[STOCK_SYMBOL] = self.tomato_trader.run(state, tomato_prices)

        # Persist tomato price history for next iteration
        traderData = jsonpickle.encode(tomato_prices)
        conversions = 0
        return result, conversions, traderData