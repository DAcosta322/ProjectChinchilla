"""Strategy 2: Tranched pepper accumulation.

Instead of buying 80 instantly, ramp linearly over the round.
target = min(POS_LIMIT, floor(ts / 100000 * POS_LIMIT * TRANCHE_RATE))
At TRANCHE_RATE=1.0: reach +80 exactly at round end.
At TRANCHE_RATE=2.0: reach +80 at half-round.
At TRANCHE_RATE=9.9: effectively instant (original behavior).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


class OsmiumParams:
    SYMBOL = "ASH_COATED_OSMIUM"
    POS_LIMIT = 80
    MA_WINDOW = 30
    ANCHOR = 10000
    ANCHOR_WEIGHT = 0.15
    HALF_SPREAD = 8
    NARROW_SPREAD = 13
    NARROW_EDGE = 1
    INV_SKEW = 2
    DRIFT_STRENGTH = 25


class PepperParams:
    SYMBOL = "INTARIAN_PEPPER_ROOT"
    POS_LIMIT = 80
    BUY_LIMIT = 13008
    ROUND_TS = 100000          # round duration in timestamp units
    TRANCHE_RATE = 1.2         # NEW: 1.0 = linear over round, 9.9 = instant


class Trader:

    def run(self, state: TradingState):
        result = {}
        prices = []
        pep_prices = []
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                prices = td.get("p", [])
                pep_prices = td.get("pp", [])
            except Exception:
                pass

        if OsmiumParams.SYMBOL in state.order_depths:
            result[OsmiumParams.SYMBOL] = self._trade_osmium(state, prices)
        if PepperParams.SYMBOL in state.order_depths:
            result[PepperParams.SYMBOL] = self._trade_pepper(state)

        return result, 0, json.dumps({"p": prices, "pp": pep_prices})

    def _trade_osmium(self, state: TradingState, prices: List[float]) -> List[Order]:
        P = OsmiumParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        mid = (best_bid + best_ask) / 2 if (best_bid is not None and best_ask is not None) else P.ANCHOR
        prices.append(mid)
        if len(prices) > P.MA_WINDOW:
            prices[:] = prices[-P.MA_WINDOW:]
        ma_fv = sum(prices) / len(prices)
        fv = round(ma_fv * (1 - P.ANCHOR_WEIGHT) + P.ANCHOR * P.ANCHOR_WEIGHT)
        target_pos = max(-P.POS_LIMIT, min(P.POS_LIMIT,
                         round(-P.DRIFT_STRENGTH * (fv - P.ANCHOR))))
        fv_eff = fv - round(P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT)

        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else 99
        buy_edge = 0
        sell_edge = 0
        if spread <= P.NARROW_SPREAD and od.buy_orders and od.sell_orders:
            bv = sum(od.buy_orders.values())
            av = sum(-v for v in od.sell_orders.values())
            if bv > av:
                buy_edge = P.NARROW_EDGE
            elif av > bv:
                sell_edge = P.NARROW_EDGE

        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fv_eff + buy_edge and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(P.SYMBOL, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fv_eff - sell_edge and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(P.SYMBOL, price, -qty))
                    sell_cap -= qty

        if best_bid is not None and best_ask is not None:
            our_bid = best_bid + 1
            our_ask = best_ask - 1
            if our_bid >= our_ask:
                our_bid = best_bid
                our_ask = best_ask
        elif best_bid is not None:
            our_bid = best_bid + 1
            our_ask = fv + P.HALF_SPREAD
        elif best_ask is not None:
            our_bid = fv - P.HALF_SPREAD
            our_ask = best_ask - 1
        else:
            our_bid = fv - P.HALF_SPREAD
            our_ask = fv + P.HALF_SPREAD

        our_bid = min(our_bid, fv_eff - 1)
        our_ask = max(our_ask, fv_eff)

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))

        return orders

    def _trade_pepper(self, state: TradingState) -> List[Order]:
        P = PepperParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        # Tranched target: ramp up linearly over the round
        ts = state.timestamp
        progress = min(1.0, ts / P.ROUND_TS)
        target = min(P.POS_LIMIT, int(progress * P.POS_LIMIT * P.TRANCHE_RATE))
        buy_cap = max(0, target - pos)

        if buy_cap <= 0:
            return orders

        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price > P.BUY_LIMIT or buy_cap <= 0:
                    break
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                buy_cap -= qty

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, P.BUY_LIMIT, buy_cap))

        return orders
