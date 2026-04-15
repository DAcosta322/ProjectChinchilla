"""ASH_COATED_OSMIUM market-making + INTARIAN_PEPPER_ROOT accumulation.

Osmium: Dual-MA fair value with market-making (unchanged from round1_osmium).
Pepper Root: Buy at/below fair value, then hold. Never sell.
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

    FAST_WINDOW = 22
    SLOW_WINDOW = 300
    SIGNAL_MULT = 0.65

    SPREAD = 3
    CLEAR_THRESHOLD = 70
    GAMMA = 0.01

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

        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fv and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fv and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

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

        if buy_capacity > 0:
            orders.append(Order(self.name, fv - self.SPREAD, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, fv + self.SPREAD, -sell_capacity))

        return orders


class PepperTrader(ProductTrader):
    """Buy pepper roots at fair value and hold. Never sell.

    Strategy: Track the mid price with a short MA to estimate fair value.
    Take any asks at or below fair value, and post a resting bid at fv - 1
    to accumulate up to the position limit.
    """

    MA_WINDOW = 50

    def __init__(self):
        super().__init__(PEPPER, fair_value=30000, pos_limit=POS_LIMITS[PEPPER])

    def compute_fair_value(self, order_depth: OrderDepth, price_history: List[float]) -> float:
        mid = self.get_mid_price(order_depth)
        price_history.append(mid)

        if len(price_history) > self.MA_WINDOW:
            price_history[:] = price_history[-self.MA_WINDOW:]

        if len(price_history) >= 5:
            return sum(price_history) / len(price_history)
        return mid

    def run(self, state: TradingState, price_history: List[float]) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        fv = round(self.compute_fair_value(order_depth, price_history))
        buy_capacity = self.pos_limit - position

        if buy_capacity <= 0:
            return orders

        # Take any asks at or below fair value
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price <= fv and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        # Post resting bid to accumulate more
        if buy_capacity > 0:
            orders.append(Order(self.name, fv - 1, buy_capacity))

        return orders


class Trader:

    def __init__(self):
        self.osmium_trader = OsmiumTrader()
        self.pepper_trader = PepperTrader()

    def run(self, state: TradingState):
        result = {}

        trader_state: dict = {}
        if state.traderData:
            try:
                trader_state = json.loads(state.traderData)
            except Exception:
                trader_state = {}

        osmium_prices: List[float] = trader_state.get("osmium_prices", [])
        pepper_prices: List[float] = trader_state.get("pepper_prices", [])

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self.osmium_trader.run(state, osmium_prices)

        if PEPPER in state.order_depths:
            result[PEPPER] = self.pepper_trader.run(state, pepper_prices)

        trader_state["osmium_prices"] = osmium_prices
        trader_state["pepper_prices"] = pepper_prices

        traderData = json.dumps(trader_state)
        conversions = 0
        return result, conversions, traderData
