"""Round 3 — Double moving average crossover for HYDROGEL_PACK and
VELVETFRUIT_EXTRACT.

Strategy: golden / death cross.
  short MA (500 ticks) > long MA (10000 ticks) -> target +POS_LIMIT
  short MA (500 ticks) < long MA (10000 ticks) -> target -POS_LIMIT

Fill style: take the book up to short_ma, then post a passive
penny-inside quote with any remaining capacity.

Warm-up: do not trade until the short MA has SHORT_WINDOW samples; the
long MA uses the partial running average in the meantime.

Long-window storage note: a true rolling 10000-window SMA would need to
keep 10000 mids per product, which exceeds the 50K traderData budget.
The final simulation is exactly 10000 iterations, so a running sum + count
matches a 10000-window SMA at every tick of that run. For longer backtests
the long MA grows past 10000 samples — accepted as a known limitation.
"""

from datamodel import TradingState, Order
from typing import List
import json


class DoubleMACrossoverTrader:
    SYMBOL: str = ""
    POS_LIMIT: int = 0
    SHORT_WINDOW: int = 500
    LONG_WINDOW: int = 10000

    def __init__(self) -> None:
        self.short_hist: List[float] = []
        self.short_sum: float = 0.0
        self.long_sum: float = 0.0
        self.long_count: int = 0

    def load(self, blob: dict) -> None:
        self.short_hist = list(blob.get("sh", []))
        self.short_sum = sum(self.short_hist)
        self.long_sum = float(blob.get("ls", 0.0))
        self.long_count = int(blob.get("lc", 0))

    def dump(self) -> dict:
        return {
            "sh": self.short_hist,
            "ls": self.long_sum,
            "lc": self.long_count,
        }

    def _update_history(self, mid: float) -> None:
        self.short_hist.append(mid)
        self.short_sum += mid
        if len(self.short_hist) > self.SHORT_WINDOW:
            self.short_sum -= self.short_hist.pop(0)
        self.long_sum += mid
        self.long_count += 1

    def run(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths.get(self.SYMBOL)
        if od is None or not od.buy_orders or not od.sell_orders:
            return orders

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        mid = (best_bid + best_ask) / 2
        self._update_history(mid)

        if len(self.short_hist) < self.SHORT_WINDOW:
            return orders

        short_ma = self.short_sum / len(self.short_hist)
        long_ma = self.long_sum / self.long_count

        if short_ma > long_ma:
            target = self.POS_LIMIT
        elif short_ma < long_ma:
            target = -self.POS_LIMIT
        else:
            return orders

        pos = state.position.get(self.SYMBOL, 0)
        delta = target - pos
        if delta == 0:
            return orders

        if delta > 0:
            cap = delta
            for price in sorted(od.sell_orders.keys()):
                if price > short_ma or cap <= 0:
                    break
                qty = min(-od.sell_orders[price], cap)
                orders.append(Order(self.SYMBOL, price, qty))
                cap -= qty
            if cap > 0:
                quote = best_bid + 1 if best_bid + 1 < best_ask else best_bid
                orders.append(Order(self.SYMBOL, quote, cap))
        else:
            cap = -delta
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price < short_ma or cap <= 0:
                    break
                qty = min(od.buy_orders[price], cap)
                orders.append(Order(self.SYMBOL, price, -qty))
                cap -= qty
            if cap > 0:
                quote = best_ask - 1 if best_ask - 1 > best_bid else best_ask
                orders.append(Order(self.SYMBOL, quote, -cap))

        return orders


class HydrogelPackTrader(DoubleMACrossoverTrader):
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    SHORT_WINDOW = 500
    LONG_WINDOW = 10000


class VelvetfruitExtractTrader(DoubleMACrossoverTrader):
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    SHORT_WINDOW = 500
    LONG_WINDOW = 10000


class Trader:

    def __init__(self) -> None:
        self.hydrogel = HydrogelPackTrader()
        self.velvet = VelvetfruitExtractTrader()

    def bid(self) -> int:
        return 0

    def run(self, state: TradingState):
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                self.hydrogel.load(td.get("hg", {}))
                self.velvet.load(td.get("ve", {}))
            except Exception:
                pass

        result = {
            self.hydrogel.SYMBOL: self.hydrogel.run(state),
            self.velvet.SYMBOL: self.velvet.run(state),
        }

        td_out = json.dumps({
            "hg": self.hydrogel.dump(),
            "ve": self.velvet.dump(),
        })
        return result, 0, td_out
