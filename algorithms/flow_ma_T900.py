"""Dual-MA + trade flow strategy.

Builds on short_warmup_ma.py, adding a trade flow signal from market_trades
to detect directional pressure before the MA can react.

FV = ma_fair_value + flow_adjustment - GAMMA * position

The flow signal is event-driven: when a market trade happens (only ~4% of
ticks), we record its direction and apply a FV shift for the next FLOW_WINDOW
ticks. After that window expires with no new trades, the shift drops to zero
and we fall back to pure MA + inventory penalty.

This avoids wasting signal strength on a decaying EMA that's mostly zero.
When a trade fires, the algo reacts at full strength immediately.

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
    """Dual-MA + event-driven trade flow strategy for TOMATOES.

    Three layers:
    1. MA baseline: slow_ma + SIGNAL_MULT * (fast_ma - slow_ma)
       Gives the general price level. Lags during fast moves.
    2. Trade flow (event-driven): when a market trade happens, record its
       direction and shift FV for the next FLOW_WINDOW ticks. No trade
       recently = no shift. Full strength when it fires, zero otherwise.
    3. Inventory penalty: -GAMMA * position.
       Prevents position drift to extremes.
    """

    # --- MA parameters (same as short_warmup_ma) ---
    FAST_WINDOW = 10
    SLOW_WINDOW = 200
    SIGNAL_MULT = 1.0

    # --- Trade flow parameters ---
    FLOW_WINDOW = 25     # Ticks to keep the flow signal active after a trade
    FLOW_MULT = 0.5      # FV shift per unit of net flow (in price points)

    # --- Risk parameters ---
    SPREAD = 3
    CLEAR_THRESHOLD = 40
    GAMMA = 0.05

    def __init__(self):
        super().__init__(STOCK_SYMBOL, fair_value=5000, pos_limit=POS_LIMITS[STOCK_SYMBOL])

    def compute_ma_fair_value(self, order_depth: OrderDepth, price_history: List[float]) -> float:
        mid = self.get_mid_price(order_depth)
        price_history.append(mid)

        if len(price_history) > self.SLOW_WINDOW:
            price_history[:] = price_history[-self.SLOW_WINDOW:]

        if len(price_history) > self.FAST_WINDOW:
            fast_ma = sum(price_history[-self.FAST_WINDOW:]) / self.FAST_WINDOW
            slow_ma = sum(price_history) / len(price_history)
            signal = fast_ma - slow_ma
            return slow_ma + signal * self.SIGNAL_MULT
        return mid

    def compute_flow_signal(self, state: TradingState, trader_state: dict, mid: float) -> float:
        """Event-driven flow: full signal when a trade happened recently, zero otherwise."""
        market_trades = state.market_trades.get(self.name, [])

        if market_trades:
            # A trade happened this tick — classify and record
            net_flow = 0.0
            for trade in market_trades:
                if trade.price >= mid:
                    net_flow += trade.quantity
                else:
                    net_flow -= trade.quantity
            trader_state["last_flow"] = net_flow
            trader_state["flow_age"] = 0
            return net_flow
        else:
            # No trade this tick — use cached signal if still fresh
            age = trader_state.get("flow_age", self.FLOW_WINDOW)
            if age < self.FLOW_WINDOW:
                trader_state["flow_age"] = age + 1
                return trader_state.get("last_flow", 0.0)
            return 0.0

    def run(self, state: TradingState, price_history: List[float], trader_state: dict) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        # Layer 1: MA baseline
        ma_fv = self.compute_ma_fair_value(order_depth, price_history)
        mid = self.get_mid_price(order_depth)

        # Layer 2: Trade flow (event-driven)
        flow = self.compute_flow_signal(state, trader_state, mid)

        # Layer 3: Inventory penalty
        adjusted_fv = ma_fv + self.FLOW_MULT * flow - self.GAMMA * position
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

        # Restore persisted state
        trader_state: dict = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        # Extract price history from state (stored as list)
        tomato_prices: List[float] = trader_state.get("prices", [])

        if COMMODITY_SYMBOL in state.order_depths:
            result[COMMODITY_SYMBOL] = self.emerald_trader.run(state)

        if STOCK_SYMBOL in state.order_depths:
            result[STOCK_SYMBOL] = self.tomato_trader.run(state, tomato_prices, trader_state)

        # Persist price history back into state
        trader_state["prices"] = tomato_prices

        traderData = json.dumps(trader_state)
        conversions = 0
        return result, conversions, traderData
