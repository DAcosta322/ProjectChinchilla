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
    MA_WINDOW = 30            # swept: +186 vs 40
    ANCHOR = 10000
    ANCHOR_WEIGHT = 0.15      # blend MA with anchor for mean-reversion pull
    HALF_SPREAD = 8           # fallback when book is one-sided
    NARROW_SPREAD = 13        # threshold for "narrow spread" detection
    NARROW_EDGE = 1           # extra FV tolerance on narrow-spread ticks
    INV_SKEW = 2              # flatten bias: shift fv by INV_SKEW*(pos-target)/POS_LIMIT
    # Mean-reversion bias: target short pos when fv>ANCHOR, long when fv<ANCHOR.
    # target_pos = -DRIFT_STRENGTH * (fv - ANCHOR), clamped to +-POS_LIMIT.
    # Swept optimum: 25 (stacks +223 with MA_WINDOW=30).
    # At drift=+-3.2 this pins target to +-POS_LIMIT.
    DRIFT_STRENGTH = 25

class PepperParams:
    SYMBOL = "INTARIAN_PEPPER_ROOT"
    POS_LIMIT = 80
    # Pepper is monotone increasing ~1pt/tick. starts at 13000 on day 2.
    # 13008 outperformed 13010 empirically (deeper trough at 13010).
    BUY_LIMIT = 13008


# =====================================================================
# Trader
# =====================================================================

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

        # MA fair value blended with anchor for mean-reversion pull
        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        else:
            mid = P.ANCHOR
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

        # Phase 1: Take mispriced orders
        # On narrow-spread ticks, widen the taking threshold on the
        # imbalance-confirmed side only (positive imb -> buy wider,
        # negative imb -> sell wider). This avoids round-trip churn.
        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else 99
        buy_edge = 0
        sell_edge = 0
        if spread <= P.NARROW_SPREAD and od.buy_orders and od.sell_orders:
            bv = sum(od.buy_orders.values())
            av = sum(-v for v in od.sell_orders.values())
            if bv > av:
                buy_edge = P.NARROW_EDGE   # more bids -> price going up -> buy wider
            elif av > bv:
                sell_edge = P.NARROW_EDGE  # more asks -> price going down -> sell wider

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

        # Phase 2: Post resting orders to capture bot flow
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

        # Clamp resting orders: bid at most fv-1, ask at least fv.
        # Prevents losing resting fills at wrong-side prices.
        our_bid = min(our_bid, fv_eff - 1)
        our_ask = max(our_ask, fv_eff)

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))

        return orders

    # ---------------------------PEPPER---------------------------------
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