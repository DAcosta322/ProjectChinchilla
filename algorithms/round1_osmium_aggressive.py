"""ASH_COATED_OSMIUM — aggressive momentum variant (~80% efficiency).

High SIGNAL_MULT (1.0) chases trends harder; high GAMMA (0.1) penalises
inventory aggressively. Over-corrects on whipsaws but profits on sustained moves.

Backtested PnL: ~9471 (80% of optimal 11779).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

POS_LIMITS = {
    OSMIUM: 80,
    PEPPER: 80,
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


class OsmiumTrader(ProductTrader):
    """MA-based market maker for ASH_COATED_OSMIUM."""

    # --- MA parameters ---
    FAST_WINDOW = 20
    SLOW_WINDOW = 300
    SIGNAL_MULT = 1.0

    # --- Risk parameters ---
    SPREAD = 3
    CLEAR_THRESHOLD = 40
    GAMMA = 0.1

    def __init__(self):
        super().__init__(OSMIUM, fair_value=10000, pos_limit=POS_LIMITS[OSMIUM])

    def compute_fair_value(self, order_depth: OrderDepth, price_history: List[float]) -> float:
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

    def run(self, state: TradingState, price_history: List[float]) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        ma_fv = self.compute_fair_value(order_depth, price_history)
        adjusted_fv = ma_fv - self.GAMMA * position
        fv = round(adjusted_fv)

        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        # Take cheap sell orders (buy below fair value)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fv and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        # Take expensive buy orders (sell above fair value)
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fv and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

        # Inventory clearing
        if position > self.CLEAR_THRESHOLD:
            clear_qty = min(position - self.CLEAR_THRESHOLD, sell_capacity)
            if clear_qty > 0:
                orders.append(Order(self.name, fv, -clear_qty))
                sell_capacity -= clear_qty
        elif position < -self.CLEAR_THRESHOLD:
            clear_qty = min(-position - self.CLEAR_THRESHOLD, buy_capacity)
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
        self.osmium_trader = OsmiumTrader()

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

        osmium_prices: List[float] = trader_state.get("prices", [])

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self.osmium_trader.run(state, osmium_prices)

        trader_state["prices"] = osmium_prices

        traderData = json.dumps(trader_state)
        conversions = 0
        return result, conversions, traderData
