"""PnL-guarded MA + flow strategy.

Builds on flow_ma.py, adding a live PnL tracker that overrides the MA signal
when the algo is losing money. PnL drawdown is a direct measure of "we're
on the wrong side" — it doesn't try to predict price direction from noisy
mid prices, it just reacts to the result.

Priority order:
  1. PnL drawdown exceeds threshold → STOP taking, aggressively flatten
  2. PnL recovering / normal        → flow_ma logic as usual

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
    """PnL-guarded MA + flow strategy for TOMATOES.

    Three states:
      "normal"      — full MA + GAMMA trading logic
      "guard_long"  — was long and got hurt. Blocks buys, unwinds longs.
      "guard_short" — was short and got hurt. Blocks sells, unwinds shorts.

    Transitions:
      normal → guard_long:  drawdown >= DD_THRESHOLD while position > 0
      normal → guard_short: drawdown >= DD_THRESHOLD while position < 0
      guard_long  → normal: position <= GUARD_EXIT (flattened)
      guard_short → normal: position >= -GUARD_EXIT (flattened)

    The directional state is sticky — once triggered by being wrong-way long,
    it stays in guard_long even if position temporarily crosses zero from
    clearing overshoot. This prevents the algo from immediately rebuilding
    the same losing position.
    """

    # --- MA parameters ---
    FAST_WINDOW = 10
    SLOW_WINDOW = 200
    SIGNAL_MULT = 1.0

    # --- Trade flow parameters ---
    FLOW_WINDOW = 25
    FLOW_MULT = 0.0      # Disabled: flow signal adds noise, not value on 4% trade density

    # --- Risk parameters ---
    SPREAD = 3
    CLEAR_THRESHOLD = 40
    GAMMA = 0.05

    # --- PnL guard parameters ---
    DD_THRESHOLD = 250   # Drawdown from peak to trigger guard mode
    GUARD_EXIT = 5       # Exit guard mode when abs(position) <= this
    GUARD_CLEAR = 5      # In guard mode, clear above this (much tighter than normal)

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
            age = trader_state.get("flow_age", self.FLOW_WINDOW)
            if age < self.FLOW_WINDOW:
                trader_state["flow_age"] = age + 1
                return trader_state.get("last_flow", 0.0)
            return 0.0

    def update_pnl(self, state: TradingState, trader_state: dict, mid: float) -> float:
        """Track live PnL from own_trades and current position mark-to-market."""
        cash = trader_state.get("cash", 0.0)

        # Process fills from last tick
        own_trades = state.own_trades.get(self.name, [])
        for trade in own_trades:
            if trade.buyer == "SUBMISSION":
                cash -= trade.price * trade.quantity
            elif trade.seller == "SUBMISSION":
                cash += trade.price * trade.quantity

        trader_state["cash"] = cash

        position = state.position.get(self.name, 0)
        live_pnl = cash + position * mid

        # Track peak PnL
        peak_pnl = trader_state.get("peak_pnl", 0.0)
        if live_pnl > peak_pnl:
            peak_pnl = live_pnl
            trader_state["peak_pnl"] = peak_pnl

        drawdown = peak_pnl - live_pnl
        return drawdown

    def run(self, state: TradingState, price_history: List[float], trader_state: dict) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        # Layer 1: MA baseline (always compute to keep history updated)
        ma_fv = self.compute_ma_fair_value(order_depth, price_history)
        mid = self.get_mid_price(order_depth)

        # PnL guard: state machine
        drawdown = self.update_pnl(state, trader_state, mid)
        guard = trader_state.get("guard", "normal")  # "normal" | "guard_long" | "guard_short"

        # Transition: normal → guard
        if guard == "normal" and drawdown >= self.DD_THRESHOLD:
            if position > 0:
                guard = "guard_long"
            elif position < 0:
                guard = "guard_short"

        # Transition: guard → normal (position flattened)
        if guard == "guard_long" and position <= self.GUARD_EXIT:
            guard = "normal"
            trader_state["peak_pnl"] = trader_state.get("cash", 0.0) + position * mid
        elif guard == "guard_short" and position >= -self.GUARD_EXIT:
            guard = "normal"
            trader_state["peak_pnl"] = trader_state.get("cash", 0.0) + position * mid

        trader_state["guard"] = guard

        if guard == "guard_long":
            # Was long and got hurt — block buys, unwind longs
            flow = self.compute_flow_signal(state, trader_state, mid)
            adjusted_fv = ma_fv + self.FLOW_MULT * flow - self.GAMMA * position
            fv = round(adjusted_fv)

            buy_capacity = self.pos_limit - position
            sell_capacity = self.pos_limit + position

            # Allow sell-taking (unwind at good prices)
            if order_depth.buy_orders:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price > fv and sell_capacity > 0:
                        bid_vol = order_depth.buy_orders[bid_price]
                        qty = min(bid_vol, sell_capacity)
                        orders.append(Order(self.name, bid_price, -qty))
                        sell_capacity -= qty

            # Aggressive clearing toward GUARD_CLEAR
            if position > self.GUARD_CLEAR:
                clear_qty = min(position - self.GUARD_CLEAR, sell_capacity)
                if clear_qty > 0:
                    orders.append(Order(self.name, fv, -clear_qty))
                    sell_capacity -= clear_qty

            # Resting sell only
            if sell_capacity > 0:
                orders.append(Order(self.name, fv + self.SPREAD, -sell_capacity))

            return orders

        if guard == "guard_short":
            # Was short and got hurt — block sells, unwind shorts
            flow = self.compute_flow_signal(state, trader_state, mid)
            adjusted_fv = ma_fv + self.FLOW_MULT * flow - self.GAMMA * position
            fv = round(adjusted_fv)

            buy_capacity = self.pos_limit - position
            sell_capacity = self.pos_limit + position

            # Allow buy-taking (unwind at good prices)
            if order_depth.sell_orders:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price < fv and buy_capacity > 0:
                        ask_vol = -order_depth.sell_orders[ask_price]
                        qty = min(ask_vol, buy_capacity)
                        orders.append(Order(self.name, ask_price, qty))
                        buy_capacity -= qty

            # Aggressive clearing toward GUARD_CLEAR
            if position < -self.GUARD_CLEAR:
                clear_qty = min(-position - self.GUARD_CLEAR, buy_capacity)
                if clear_qty > 0:
                    orders.append(Order(self.name, fv, clear_qty))
                    buy_capacity -= clear_qty

            # Resting buy only
            if buy_capacity > 0:
                orders.append(Order(self.name, fv - self.SPREAD, buy_capacity))

            return orders

        # NORMAL MODE: flow_ma logic
        flow = self.compute_flow_signal(state, trader_state, mid)
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
        filled_buy = self.pos_limit - (state.position.get(self.name, 0)) - buy_capacity
        filled_sell = self.pos_limit + (state.position.get(self.name, 0)) - sell_capacity
        current_pos = state.position.get(self.name, 0) + filled_buy - filled_sell

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

        trader_state: dict = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        tomato_prices: List[float] = trader_state.get("prices", [])

        if COMMODITY_SYMBOL in state.order_depths:
            result[COMMODITY_SYMBOL] = self.emerald_trader.run(state)

        if STOCK_SYMBOL in state.order_depths:
            result[STOCK_SYMBOL] = self.tomato_trader.run(state, tomato_prices, trader_state)

        trader_state["prices"] = tomato_prices

        traderData = json.dumps(trader_state)
        conversions = 0
        return result, conversions, traderData
