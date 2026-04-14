"""EMA-based trend-following strategy.

Copy of tutorial.py with the TOMATOES StockTrader replaced by an EMA-based
approach. EMA has ZERO warm-up — it produces a signal from tick 1.

Uses fast EMA vs slow EMA crossover to detect trends:
- When fast > slow (bullish): fair value shifts up -> more willing to buy
- When fast < slow (bearish): fair value shifts down -> more willing to sell
- Signal strength scales with the gap between EMAs

All EMERALDS logic identical to tutorial.py.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

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
    """EMA trend-following strategy for TOMATOES.

    Uses exponential moving averages which have no warm-up period.
    The EMA updates every tick, so the signal is available from the start.

    Fair value = slow_ema + SIGNAL_MULT * (fast_ema - slow_ema)
    When fast_ema > slow_ema: bullish, FV shifts up
    When fast_ema < slow_ema: bearish, FV shifts down
    """

    # --- Tunable parameters (overridden by param_sweep) ---
    FAST_ALPHA = 0.25    # EMA smoothing for fast (higher = more responsive)
    SLOW_ALPHA = 0.02    # EMA smoothing for slow (lower = smoother)
    SIGNAL_MULT = 1.5    # How aggressively to follow the trend
    SPREAD = 2           # Resting order distance from FV
    CLEAR_THRESHOLD = 65 # Start inventory clearing above this

    def __init__(self):
        super().__init__(STOCK_SYMBOL, fair_value=5000, pos_limit=POS_LIMITS[STOCK_SYMBOL])

    def compute_fair_value(self, order_depth: OrderDepth, ema_state: dict) -> float:
        mid = self.get_mid_price(order_depth)

        fast_ema = ema_state.get("fast_ema")
        slow_ema = ema_state.get("slow_ema")

        if fast_ema is None:
            # First tick: initialize both EMAs to mid
            ema_state["fast_ema"] = mid
            ema_state["slow_ema"] = mid
            return mid

        # Update EMAs
        fast_ema = self.FAST_ALPHA * mid + (1 - self.FAST_ALPHA) * fast_ema
        slow_ema = self.SLOW_ALPHA * mid + (1 - self.SLOW_ALPHA) * slow_ema

        ema_state["fast_ema"] = fast_ema
        ema_state["slow_ema"] = slow_ema

        # Signal: fast - slow crossover
        signal = fast_ema - slow_ema
        return slow_ema + signal * self.SIGNAL_MULT

    def run(self, state: TradingState, ema_state: dict) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        fair_value = self.compute_fair_value(order_depth, ema_state)
        fv = round(fair_value)

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

        # Restore EMA state
        ema_state: dict = {}
        if state.traderData:
            try:
                ema_state = json.loads(state.traderData)
            except Exception:
                ema_state = {}

        if COMMODITY_SYMBOL in state.order_depths:
            result[COMMODITY_SYMBOL] = self.emerald_trader.run(state)

        if STOCK_SYMBOL in state.order_depths:
            result[STOCK_SYMBOL] = self.tomato_trader.run(state, ema_state)

        traderData = json.dumps(ema_state)
        conversions = 0
        return result, conversions, traderData
