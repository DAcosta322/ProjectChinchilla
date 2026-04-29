"""Round 5 — PEBBLES-only trader.

Two layers:

1. Per-product MM with momentum tilt.
   - Anchor each PEBBLES_* product to its mid (the basket constraint pins
     sum(mids)=50000 essentially exactly, so mid is always the right
     fair value at the level layer).
   - Track a fast EMA of mid velocity. Tilt the anchor by MOM_GAIN*velocity
     so quotes lean with momentum.
   - Inventory skew: shift anchor by -INV_SKEW * (pos / POS_LIMIT).
   - Aggressive cross when best bid/ask is mispriced beyond AGG_EDGE.
   - Passive MM at anchor ± MM_EDGE, sized at MM_QTY.

2. Basket ETF arb on bid/ask deviations.
   - Invariant: PEBBLES_XS + S + M + L + XL = 50000 (R²≈1.0, regression
     intercept 50000±2 across days 2/3/4, see project_round5_baskets memory).
   - When sum(best_bids) > 50000+EDGE: short the ETF — sell each leg at
     its bid for a locked profit per basket.
   - When sum(best_asks) < 50000-EDGE: long the ETF — buy each leg at its
     ask for locked profit.
   - Quantity is capped by min(top-of-book volume, remaining capacity)
     across all 5 legs simultaneously.

State persisted via traderData JSON: per-product mid history (deque of
recent mids) for momentum EMA.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


class P:
    PRODUCTS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    BASKET_SUM = 50000
    POS_LIMIT = 10

    # Momentum (per-product velocity tilt)
    MOM_SPAN = 30          # EMA span for velocity smoothing
    MOM_GAIN = 0.5         # FV adjustment per unit velocity (price/tick)

    # Per-product MM
    INV_SKEW = 4.0         # FV shift per unit (pos / POS_LIMIT). 4 ≈ half a tight spread
    AGG_EDGE = 3.0         # cross spread when mid-price > anchor + AGG_EDGE
    MM_EDGE = 1            # post passive bid/ask at anchor ± MM_EDGE
    MM_QTY = 5             # passive quote size

    # Basket ETF arb
    BASKET_ARB_EDGE = 0    # take any non-negative edge (always lock profit if sum allows)
    BASKET_ARB_QTY = 10    # max baskets per tick (= POS_LIMIT)


def best_bid_ask(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None, None, 0, 0
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


class Trader:
    def run(self, state: TradingState):
        # Load state
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}
        # Per-product velocity EMA (price units per tick)
        vel = data.get("vel", {p: 0.0 for p in P.PRODUCTS})
        last_mid = data.get("last_mid", {})

        # Snapshot mids and book
        mids: Dict[str, float] = {}
        books: Dict[str, tuple] = {}
        for p in P.PRODUCTS:
            od = state.order_depths.get(p)
            bb, ba, bv, av = best_bid_ask(od)
            if bb is None:
                continue
            mids[p] = (bb + ba) / 2.0
            books[p] = (bb, ba, bv, av)

        if len(mids) != 5:
            # Bail if any leg is missing — basket arb requires all 5
            data_out = {"vel": vel, "last_mid": {p: mids[p] for p in mids}}
            return {}, 0, json.dumps(data_out)

        # Update velocity EMA
        alpha = 2.0 / (P.MOM_SPAN + 1.0)
        for p in P.PRODUCTS:
            if p in last_mid:
                d = mids[p] - last_mid[p]
                vel[p] = (1 - alpha) * vel.get(p, 0.0) + alpha * d

        orders: Dict[str, List[Order]] = {p: [] for p in P.PRODUCTS}

        # Track in-flight order qty per product to enforce position limits
        # against the algo's own combined orders this tick.
        buy_used = {p: 0 for p in P.PRODUCTS}   # qty intended to buy
        sell_used = {p: 0 for p in P.PRODUCTS}  # qty intended to sell

        def buy_capacity(p):
            return P.POS_LIMIT - state.position.get(p, 0) - buy_used[p]

        def sell_capacity(p):
            return P.POS_LIMIT + state.position.get(p, 0) - sell_used[p]

        # ---- Layer 1: per-product MM with momentum + inventory skew ----
        for p in P.PRODUCTS:
            bb, ba, bv, av = books[p]
            pos = state.position.get(p, 0)

            # Anchor: mid + momentum tilt - inventory skew
            anchor = mids[p] + P.MOM_GAIN * vel[p] - P.INV_SKEW * (pos / P.POS_LIMIT)

            # --- Aggressive cross (take liquidity at extreme misprice) ---
            if ba < anchor - P.AGG_EDGE:
                cap = min(av, buy_capacity(p))
                if cap > 0:
                    orders[p].append(Order(p, ba, cap))
                    buy_used[p] += cap
            if bb > anchor + P.AGG_EDGE:
                cap = min(bv, sell_capacity(p))
                if cap > 0:
                    orders[p].append(Order(p, bb, -cap))
                    sell_used[p] += cap

            # --- Passive MM around anchor ---
            mm_bid_px = int(anchor - P.MM_EDGE)         # floor toward bid
            mm_ask_px = int(anchor + P.MM_EDGE) + 1     # ceil toward ask
            # Don't post inside the wrong side of the book
            if mm_bid_px < ba:
                cap = min(P.MM_QTY, buy_capacity(p))
                if cap > 0:
                    orders[p].append(Order(p, mm_bid_px, cap))
                    buy_used[p] += cap
            if mm_ask_px > bb:
                cap = min(P.MM_QTY, sell_capacity(p))
                if cap > 0:
                    orders[p].append(Order(p, mm_ask_px, -cap))
                    sell_used[p] += cap

        # ---- Layer 2: basket ETF arb at bid/ask ----
        sum_bb = sum(books[p][0] for p in P.PRODUCTS)
        sum_ba = sum(books[p][1] for p in P.PRODUCTS)

        if sum_bb > P.BASKET_SUM + P.BASKET_ARB_EDGE:
            # Sum of bids exceeds basket value — sell each leg at its bid.
            qty = P.BASKET_ARB_QTY
            for p in P.PRODUCTS:
                bb, _, bv, _ = books[p]
                qty = min(qty, bv, max(0, sell_capacity(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    bb = books[p][0]
                    orders[p].append(Order(p, bb, -qty))
                    sell_used[p] += qty

        if sum_ba < P.BASKET_SUM - P.BASKET_ARB_EDGE:
            # Sum of asks below basket value — buy each leg at its ask.
            qty = P.BASKET_ARB_QTY
            for p in P.PRODUCTS:
                _, ba, _, av = books[p]
                qty = min(qty, av, max(0, buy_capacity(p)))
            if qty > 0:
                for p in P.PRODUCTS:
                    ba = books[p][1]
                    orders[p].append(Order(p, ba, qty))
                    buy_used[p] += qty

        # Drop empty product lists
        orders = {p: o for p, o in orders.items() if o}

        # Persist state
        data_out = {"vel": vel, "last_mid": {p: mids[p] for p in P.PRODUCTS}}
        return orders, 0, json.dumps(data_out)
