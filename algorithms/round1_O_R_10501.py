"""Osmium wall-tracking market maker + Pepper Root accumulation."""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json


# =====================================================================
# Parameters
# =====================================================================

class OsmiumParams:
    SYMBOL = "ASH_COATED_OSMIUM"
    POS_LIMIT = 80
    MA_WINDOW = 20
    HALF_SPREAD = 7          # fallback when book is one-sided
    ANCHOR = 10000            # fallback mid when book is empty

class PepperParams:
    SYMBOL = "INTARIAN_PEPPER_ROOT"
    POS_LIMIT = 80
    BUY_LIMIT = 12008         # buy everything at or below this price


# =====================================================================
# Trader
# =====================================================================

class Trader:

    def run(self, state: TradingState):
        result = {}

        prices = []
        if state.traderData:
            try:
                prices = json.loads(state.traderData).get("p", [])
            except Exception:
                prices = []

        if OsmiumParams.SYMBOL in state.order_depths:
            result[OsmiumParams.SYMBOL] = self._trade_osmium(state, prices)

        if PepperParams.SYMBOL in state.order_depths:
            result[PepperParams.SYMBOL] = self._trade_pepper(state)

        return result, 0, json.dumps({"p": prices})

    # ------------------------------------------------------------------
    # OSMIUM
    # ------------------------------------------------------------------
    def _trade_osmium(self, state: TradingState, prices: List[float]) -> List[Order]:
        P = OsmiumParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # MA fair value
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        else:
            mid = P.ANCHOR
        prices.append(mid)
        if len(prices) > P.MA_WINDOW:
            prices[:] = prices[-P.MA_WINDOW:]
        fv = round(sum(prices) / len(prices))

        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        # Phase 1: Take mispriced orders
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fv and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(P.SYMBOL, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fv and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(P.SYMBOL, price, -qty))
                    sell_cap -= qty

        # Phase 2: Post just inside bot walls
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

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))

        return orders

    # ------------------------------------------------------------------
    # PEPPER
    # ------------------------------------------------------------------
    def _trade_pepper(self, state: TradingState) -> List[Order]:
        P = PepperParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        buy_cap = P.POS_LIMIT - pos

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
