"""Imbalance-gated MA strategy.

Refines tutorial_T1800.py based on the discovery that book imbalance has
-0.88 correlation with next-tick mid change on the 7% of ticks where it
fires. Verified at 97% accuracy on submission 54133.

Key insight: trades only fill on narrow-spread ticks (~7%) when a third
participant posts inside the bot walls. On these ticks, the imbalance
tells us which direction mid will revert:
  - Negative imbalance (tight ask) → mid will rise  → BUY
  - Positive imbalance (tight bid) → mid will drop  → SELL
  - Zero imbalance (normal spread) → no edge, don't take

On neutral ticks (93%), the MA lag causes bad fills (the 54133/97620 bug).
By gating takes on imbalance, we only trade when we have directional
confirmation.

Also includes:
  - Inventory penalty (GAMMA) from short_warmup_ma
  - PnL drawdown guard from pnl_guard

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
    """Imbalance-gated MA strategy for TOMATOES.

    Only takes when book imbalance confirms direction:
      - imbalance < 0 (tight ask): allow buys only
      - imbalance > 0 (tight bid): allow sells only
      - imbalance = 0 (normal spread): no taking, only clearing/resting

    FV still comes from the dual-MA for price level estimation,
    but the imbalance gates WHEN to act on it.
    """

    # --- MA parameters ---
    FAST_WINDOW = 10
    SLOW_WINDOW = 200
    SIGNAL_MULT = 1.0

    # --- Risk parameters ---
    SPREAD = 3
    CLEAR_THRESHOLD = 40
    GAMMA = 0.05

    # --- PnL guard ---
    DD_THRESHOLD = 250
    GUARD_EXIT = 5
    GUARD_CLEAR = 5

    def __init__(self):
        super().__init__(STOCK_SYMBOL, fair_value=5000, pos_limit=POS_LIMITS[STOCK_SYMBOL])

    def compute_imbalance(self, order_depth: OrderDepth) -> float:
        """Book imbalance: (bid_vol - ask_vol) / total. Nonzero on ~7% of ticks."""
        bid_vol = sum(order_depth.buy_orders.values())
        ask_vol = sum(-v for v in order_depth.sell_orders.values())
        total = bid_vol + ask_vol
        if total == 0:
            return 0.0
        return (bid_vol - ask_vol) / total

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

    def update_pnl(self, state: TradingState, trader_state: dict, mid: float) -> float:
        """Track live PnL, return drawdown from peak."""
        cash = trader_state.get("cash", 0.0)
        own_trades = state.own_trades.get(self.name, [])
        for trade in own_trades:
            if trade.buyer == "SUBMISSION":
                cash -= trade.price * trade.quantity
            elif trade.seller == "SUBMISSION":
                cash += trade.price * trade.quantity
        trader_state["cash"] = cash

        position = state.position.get(self.name, 0)
        live_pnl = cash + position * mid

        peak_pnl = trader_state.get("peak_pnl", 0.0)
        if live_pnl > peak_pnl:
            peak_pnl = live_pnl
            trader_state["peak_pnl"] = peak_pnl

        return peak_pnl - live_pnl

    def run(self, state: TradingState, price_history: List[float], trader_state: dict) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        # Always compute MA to keep history updated
        ma_fv = self.compute_ma_fair_value(order_depth, price_history)
        mid = self.get_mid_price(order_depth)
        imbalance = self.compute_imbalance(order_depth)

        # Inventory-adjusted FV
        adjusted_fv = ma_fv - self.GAMMA * position
        fv = round(adjusted_fv)

        # PnL guard state machine
        drawdown = self.update_pnl(state, trader_state, mid)
        guard = trader_state.get("guard", "normal")

        if guard == "normal" and drawdown >= self.DD_THRESHOLD:
            if position > 0:
                guard = "guard_long"
            elif position < 0:
                guard = "guard_short"

        if guard == "guard_long" and position <= self.GUARD_EXIT:
            guard = "normal"
            trader_state["peak_pnl"] = trader_state.get("cash", 0.0) + position * mid
        elif guard == "guard_short" and position >= -self.GUARD_EXIT:
            guard = "normal"
            trader_state["peak_pnl"] = trader_state.get("cash", 0.0) + position * mid

        trader_state["guard"] = guard

        # --- GUARD MODE: block taking on losing side, clear aggressively ---
        if guard == "guard_long":
            buy_capacity = self.pos_limit - position
            sell_capacity = self.pos_limit + position

            if order_depth.buy_orders:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price > fv and sell_capacity > 0:
                        qty = min(order_depth.buy_orders[bid_price], sell_capacity)
                        orders.append(Order(self.name, bid_price, -qty))
                        sell_capacity -= qty

            if position > self.GUARD_CLEAR:
                clear_qty = min(position - self.GUARD_CLEAR, sell_capacity)
                if clear_qty > 0:
                    orders.append(Order(self.name, fv, -clear_qty))
                    sell_capacity -= clear_qty

            if sell_capacity > 0:
                orders.append(Order(self.name, fv + self.SPREAD, -sell_capacity))
            return orders

        if guard == "guard_short":
            buy_capacity = self.pos_limit - position
            sell_capacity = self.pos_limit + position

            if order_depth.sell_orders:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price < fv and buy_capacity > 0:
                        qty = min(-order_depth.sell_orders[ask_price], buy_capacity)
                        orders.append(Order(self.name, ask_price, qty))
                        buy_capacity -= qty

            if position < -self.GUARD_CLEAR:
                clear_qty = min(-position - self.GUARD_CLEAR, buy_capacity)
                if clear_qty > 0:
                    orders.append(Order(self.name, fv, clear_qty))
                    buy_capacity -= clear_qty

            if buy_capacity > 0:
                orders.append(Order(self.name, fv - self.SPREAD, buy_capacity))
            return orders

        # --- NORMAL MODE: imbalance-gated taking ---
        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        if imbalance < -0.001:
            # Tight ask detected → mid will revert UP → BUY
            if order_depth.sell_orders:
                for ask_price in sorted(order_depth.sell_orders.keys()):
                    if ask_price < fv and buy_capacity > 0:
                        ask_vol = -order_depth.sell_orders[ask_price]
                        qty = min(ask_vol, buy_capacity)
                        orders.append(Order(self.name, ask_price, qty))
                        buy_capacity -= qty

        elif imbalance > 0.001:
            # Tight bid detected → mid will revert DOWN → SELL
            if order_depth.buy_orders:
                for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if bid_price > fv and sell_capacity > 0:
                        bid_vol = order_depth.buy_orders[bid_price]
                        qty = min(bid_vol, sell_capacity)
                        orders.append(Order(self.name, bid_price, -qty))
                        sell_capacity -= qty

        # else: imbalance == 0 → no taking, just clearing + resting

        # Inventory clearing (always active)
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

        # Resting orders
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
