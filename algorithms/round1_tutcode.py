from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import jsonpickle
import math

### General ### General ### General ### General ### General ###

COMMODITY_SYMBOL = "EMERALDS"
STOCK_SYMBOL = "TOMATOES"
OPTION_SYMBOL = ""
ETF_SYMBOL = ""

POS_LIMITS = {
    COMMODITY_SYMBOL: 80,
    STOCK_SYMBOL: 80,
    OPTION_SYMBOL: 0,
    ETF_SYMBOL: 0
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

        EMERALDS_DIFF=7
        # Post resting orders to capture spread on remaining capacity
        if buy_capacity > 0:
            orders.append(Order(self.name, self.fair_value - EMERALDS_DIFF, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, self.fair_value + EMERALDS_DIFF, -sell_capacity))

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
### Option ### Option ### Option ### Option ### Option ###

class OptionTrader(ProductTrader):
    """Trades options on stocks/commodities using Black-Scholes pricing.

    Estimates implied volatility from the underlying's price history,
    computes BS theoretical value, and trades against mispriced options.
    """

    # BS parameters
    RISK_FREE_RATE = 0.0      # Competition typically assumes 0
    VOL_WINDOW = 100          # Lookback window for historical vol estimate
    MIN_VOL = 0.05            # Floor on volatility to avoid degenerate prices
    EDGE_THRESHOLD = 1.0      # Min mispricing (in price units) to trade on

    def __init__(self, option_symbol: str, underlying_symbol: str,
                 strike: float, expiry_ticks: int, pos_limit: int,
                 is_call: bool = True):
        super().__init__(option_symbol, fair_value=0, pos_limit=pos_limit)
        self.underlying_symbol = underlying_symbol
        self.strike = strike
        self.expiry_ticks = expiry_ticks  # Total ticks until expiry
        self.is_call = is_call

    # --- Black-Scholes math (no scipy needed) ---

    @staticmethod
    def norm_cdf(x: float) -> float:
        """Cumulative standard normal distribution (Abramowitz & Stegun approx)."""
        a1, a2, a3, a4, a5 = (
            0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        )
        p = 0.3275911
        sign = 1.0 if x >= 0 else -1.0
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        poly = ((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t
        y = 1.0 - poly * math.exp(-x * x / 2.0)
        return 0.5 * (1.0 + sign * y)

    @staticmethod
    def norm_pdf(x: float) -> float:
        return math.exp(-x * x / 2.0) / math.sqrt(2.0 * math.pi)

    def time_to_expiry(self, timestamp: int) -> float:
        """Fraction of total duration remaining, floored at a small epsilon."""
        remaining = max(self.expiry_ticks - timestamp, 1)
        return remaining / self.expiry_ticks

    def estimate_volatility(self, price_history: List[float]) -> float:
        """Annualised volatility from log-returns over the lookback window."""
        window = price_history[-self.VOL_WINDOW:]
        if len(window) < 2:
            return self.MIN_VOL
        log_returns = [
            math.log(window[i] / window[i - 1])
            for i in range(1, len(window))
            if window[i - 1] > 0 and window[i] > 0
        ]
        if len(log_returns) < 2:
            return self.MIN_VOL
        mean = sum(log_returns) / len(log_returns)
        var = sum((r - mean) ** 2 for r in log_returns) / (len(log_returns) - 1)
        # Scale per-tick vol to "annualised" (full-game) vol
        vol = math.sqrt(var) * math.sqrt(self.expiry_ticks)
        return max(vol, self.MIN_VOL)

    def bs_price(self, S: float, K: float, T: float, sigma: float, r: float
                 ) -> float:
        """Black-Scholes European option price."""
        if T <= 0 or sigma <= 0:
            # At expiry: intrinsic value
            if self.is_call:
                return max(S - K, 0.0)
            return max(K - S, 0.0)

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        if self.is_call:
            return S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        return K * math.exp(-r * T) * self.norm_cdf(-d2) - S * self.norm_cdf(-d1)

    def bs_greeks(self, S: float, K: float, T: float, sigma: float, r: float
                  ) -> Dict[str, float]:
        """Compute delta, gamma, vega, theta for position sizing / risk."""
        if T <= 0 or sigma <= 0:
            delta = 1.0 if (self.is_call and S > K) else (-1.0 if (not self.is_call and S < K) else 0.0)
            return {"delta": delta, "gamma": 0.0, "vega": 0.0, "theta": 0.0}

        sqrt_T = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        nd1 = self.norm_cdf(d1)
        pdf_d1 = self.norm_pdf(d1)

        gamma = pdf_d1 / (S * sigma * sqrt_T)
        vega = S * pdf_d1 * sqrt_T

        if self.is_call:
            delta = nd1
            theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                     - r * K * math.exp(-r * T) * self.norm_cdf(d2))
        else:
            delta = nd1 - 1.0
            theta = (-(S * pdf_d1 * sigma) / (2.0 * sqrt_T)
                     + r * K * math.exp(-r * T) * self.norm_cdf(-d2))

        return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}

    def run(self, state: TradingState, underlying_prices: List[float]) -> List[Order]:
        orders: List[Order] = []
        order_depth = self.get_order_depth(state)
        position = self.get_position(state)

        # Need underlying price to value the option
        underlying_depth = state.order_depths.get(self.underlying_symbol)
        if not underlying_depth or not underlying_depth.buy_orders or not underlying_depth.sell_orders:
            return orders

        S = (max(underlying_depth.buy_orders.keys())
             + min(underlying_depth.sell_orders.keys())) / 2.0

        # Track underlying price history for vol estimation
        underlying_prices.append(S)
        if len(underlying_prices) > self.VOL_WINDOW:
            underlying_prices[:] = underlying_prices[-self.VOL_WINDOW:]

        T = self.time_to_expiry(state.timestamp)
        sigma = self.estimate_volatility(underlying_prices)
        fair = self.bs_price(S, self.strike, T, sigma, self.RISK_FREE_RATE)
        fair_rounded = round(fair)
        greeks = self.bs_greeks(S, self.strike, T, sigma, self.RISK_FREE_RATE)

        buy_capacity = self.pos_limit - position
        sell_capacity = self.pos_limit + position

        # Scale edge threshold by delta — deeper ITM options need wider edge
        edge = self.EDGE_THRESHOLD * max(abs(greeks["delta"]), 0.1)

        # Take underpriced sells (buy cheap options)
        if order_depth.sell_orders:
            for ask_price in sorted(order_depth.sell_orders.keys()):
                if ask_price < fair - edge and buy_capacity > 0:
                    ask_vol = -order_depth.sell_orders[ask_price]
                    qty = min(ask_vol, buy_capacity)
                    orders.append(Order(self.name, ask_price, qty))
                    buy_capacity -= qty

        # Take overpriced buys (sell expensive options)
        if order_depth.buy_orders:
            for bid_price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if bid_price > fair + edge and sell_capacity > 0:
                    bid_vol = order_depth.buy_orders[bid_price]
                    qty = min(bid_vol, sell_capacity)
                    orders.append(Order(self.name, bid_price, -qty))
                    sell_capacity -= qty

        # Post resting orders around BS fair value
        spread = max(2, round(2.0 / max(abs(greeks["delta"]), 0.05)))
        if buy_capacity > 0:
            orders.append(Order(self.name, fair_rounded - spread, buy_capacity))
        if sell_capacity > 0:
            orders.append(Order(self.name, fair_rounded + spread, -sell_capacity))

        return orders


### Option ### Option ### Option ### Option ### Option ###
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
