"""Osmium wall-tracking market maker + Pepper Root accumulation.

Osmium: unchanged from round1_osmium_wallmake.py.
Pepper: buy at/below MA fair value, hold, never sell (from round1_osmium_pepper.py).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json

OSMIUM = "ASH_COATED_OSMIUM"
PEPPER = "INTARIAN_PEPPER_ROOT"

POS_LIMIT_OSM = 80
POS_LIMIT_PEP = 80

MA_WINDOW = 20


class Trader:

    def run(self, state: TradingState):
        result = {}

        prices = []
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                prices = td.get("p", [])
                pep_prices = td.get("pp", [])
            except Exception:
                td = {}
                prices = []
                pep_prices = []
        else:
            td = {}
            pep_prices = []

        if OSMIUM in state.order_depths:
            result[OSMIUM] = self._trade_osmium(state, prices)

        if PEPPER in state.order_depths:
            result[PEPPER] = self._trade_pepper(state)

        return result, 0, json.dumps({"p": prices, "pp": pep_prices})

    # ------------------------------------------------------------------
    # OSMIUM — identical to round1_osmium_wallmake.py
    # ------------------------------------------------------------------
    def _trade_osmium(self, state: TradingState, prices: List[float]) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[OSMIUM]
        pos = state.position.get(OSMIUM, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        # Compute MA fair value
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        else:
            mid = 10000
        prices.append(mid)
        if len(prices) > MA_WINDOW:
            prices[:] = prices[-MA_WINDOW:]
        fv = round(sum(prices) / len(prices))

        buy_cap = POS_LIMIT_OSM - pos
        sell_cap = POS_LIMIT_OSM + pos

        # --- Phase 1: Take mispriced orders vs MA fair value ---
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price < fv and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(OSMIUM, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders.keys(), reverse=True):
                if price > fv and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(OSMIUM, price, -qty))
                    sell_cap -= qty

        # --- Phase 2: Post just inside bot walls to capture flow ---
        if best_bid is not None and best_ask is not None:
            our_bid = best_bid + 1
            our_ask = best_ask - 1

            # Don't cross: if spread is already tight, fall back to walls
            if our_bid >= our_ask:
                our_bid = best_bid
                our_ask = best_ask
        else:
            our_bid = fv - 8
            our_ask = fv + 8

        if buy_cap > 0:
            orders.append(Order(OSMIUM, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(OSMIUM, our_ask, -sell_cap))

        return orders

    # ------------------------------------------------------------------
    # PEPPER — buy everything below 12008, hold forever
    # ------------------------------------------------------------------
    def _trade_pepper(self, state: TradingState) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[PEPPER]
        pos = state.position.get(PEPPER, 0)
        buy_cap = POS_LIMIT_PEP - pos

        if buy_cap <= 0:
            return orders

        buy_limit = 12008

        # Take all asks at or below 12008
        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price > buy_limit or buy_cap <= 0:
                    break
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(PEPPER, price, qty))
                buy_cap -= qty

        # Resting bid at 12008 for remaining
        if buy_cap > 0:
            orders.append(Order(PEPPER, buy_limit, buy_cap))

        return orders
