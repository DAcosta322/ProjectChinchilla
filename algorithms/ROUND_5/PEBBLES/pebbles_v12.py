"""PEBBLES v12 — v11 with MIN_SPREAD_L=12 (now ALL 5 legs gated).

Sweep showed L also benefits from spread-gating at threshold 12.
L leg PnL: $7,824 → $9,673 (+$1.8K).
Total: $121,068 (v11) → $122,896 (v12). +$1,828.

All 5 legs now have MIN_SPREAD set above their typical regime, making
each leg arb-driven most of the time and selectively MM in wide-spread
windows.

Per-leg at v12: XS +$6.3K, S +$15.9K, M +$14.8K, L +$9.7K, XL +$76.2K.
Total: $122,896 (+$73K, +147% vs v1).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
PEBBLES_OTHERS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
PEBBLES_BASKET_SUM = 50000


class P:
    POS_LIMIT = 10
    MM_QTY = 7
    DEV_GAIN_XS = 25.0  # NEW: saturated dev on XS (works now with spread-gate)
    DEV_GAIN_S = 10.0   # tuned down from 25 (with XS dev now active)
    DEV_GAIN_M = 0.0
    DEV_GAIN_L = 0.0
    DEV_GAIN_XL = 25.0
    BASKET_ARB_QTY = 10
    # Spread-conditional MM thresholds (skip MM when spread <= MIN_SPREAD)
    # Sweep winners: XS=8 (XS spread mean 9.7), M=14 (M mean 13.1)
    MIN_SPREAD_XS = 8
    MIN_SPREAD_S = 2
    MIN_SPREAD_M = 14
    MIN_SPREAD_L = 12  # NEW: gate L (sweep winner)
    MIN_SPREAD_XL = 20


def _book(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


def _gain_for(p):
    return {
        "PEBBLES_XS": P.DEV_GAIN_XS,
        "PEBBLES_S":  P.DEV_GAIN_S,
        "PEBBLES_M":  P.DEV_GAIN_M,
        "PEBBLES_L":  P.DEV_GAIN_L,
        "PEBBLES_XL": P.DEV_GAIN_XL,
    }[p]


def _min_spread_for(p):
    return {
        "PEBBLES_XS": P.MIN_SPREAD_XS,
        "PEBBLES_S":  P.MIN_SPREAD_S,
        "PEBBLES_M":  P.MIN_SPREAD_M,
        "PEBBLES_L":  P.MIN_SPREAD_L,
        "PEBBLES_XL": P.MIN_SPREAD_XL,
    }[p]


class Trader:
    def run(self, state: TradingState):
        orders: Dict[str, List[Order]] = {}
        buy_used: Dict[str, int] = {}
        sell_used: Dict[str, int] = {}

        def add(p, px, qty):
            orders.setdefault(p, []).append(Order(p, px, qty))
            if qty > 0:
                buy_used[p] = buy_used.get(p, 0) + qty
            else:
                sell_used[p] = sell_used.get(p, 0) - qty

        def buy_cap(p):
            return P.POS_LIMIT - state.position.get(p, 0) - buy_used.get(p, 0)

        def sell_cap(p):
            return P.POS_LIMIT + state.position.get(p, 0) - sell_used.get(p, 0)

        def pull_to_zero_mm(p, books_map, phantom_offset=0.0):
            r = books_map.get(p)
            if r is None:
                return
            bb, ba, _, _ = r
            if ba <= bb:
                return
            spread = ba - bb
            if spread <= _min_spread_for(p):
                return  # skip MM in narrow spread
            real_pos = state.position.get(p, 0)
            phantom = real_pos + phantom_offset
            mm_bid = bb + 1 if bb + 1 < ba else None
            mm_ask = ba - 1 if ba - 1 > bb else None
            mm_qty = P.MM_QTY
            if phantom <= 0 and mm_bid is not None:
                cap = min(mm_qty, buy_cap(p))
                if cap > 0:
                    add(p, mm_bid, cap)
            if phantom >= 0 and mm_ask is not None:
                cap = min(mm_qty, sell_cap(p))
                if cap > 0:
                    add(p, mm_ask, -cap)

        peb_books = {}
        peb_mids = {}
        for p in PEBBLES:
            r = _book(state.order_depths.get(p))
            if r is not None:
                peb_books[p] = r
                peb_mids[p] = (r[0] + r[1]) / 2.0

        if len(peb_books) == 5:
            sum_others_mids = sum(peb_mids[p] for p in PEBBLES_OTHERS)
            implied_XL = PEBBLES_BASKET_SUM - sum_others_mids
            dev_XL = peb_mids["PEBBLES_XL"] - implied_XL
            for p in PEBBLES:
                offset = _gain_for(p) * dev_XL
                pull_to_zero_mm(p, peb_books, phantom_offset=offset)
            sum_ba = sum(peb_books[p][1] for p in PEBBLES)
            if sum_ba < PEBBLES_BASKET_SUM:
                qty = P.BASKET_ARB_QTY
                for p in PEBBLES:
                    qty = min(qty, peb_books[p][3], max(0, buy_cap(p)))
                if qty > 0:
                    for p in PEBBLES:
                        add(p, peb_books[p][1], qty)

        return orders, 0, ""
