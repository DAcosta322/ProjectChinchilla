"""Round 2 — 0418b build.

Copy of 0418 + URGENCY (proven live winner in 298604, +141 PnL).
Reverted from 0419 which regressed live (-400): BUY_MARGIN=5 overpaid on
pepper (-225) and DRIFT=40 alone couldn't replace URGENCY's reactive-flow
capture (-175 osmium).

Only difference vs 0418: URGENCY widens take thresholds when pos is far
from target_pos, letting us capture bot aggressive flow during drift
windows (e.g., the ts=18500-22000 short window observed in 288855).
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
    MICROPRICE_WEIGHT = 1.0   # 0=plain mid, 1=pure microprice
    URGENCY_EDGE = 3          # widens take by this many ticks per urgency unit
    URGENCY_THRESHOLD = 0.5   # only activate when |pos-target|/POS_LIMIT > this


class PepperParams:
    SYMBOL = "INTARIAN_PEPPER_ROOT"
    POS_LIMIT = 80
    BUY_MARGIN = 1            # cap = first_observed_ask + BUY_MARGIN


class Trader:

    def run(self, state: TradingState):
        result = {}
        prices = []
        pep_first_ask = None
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                prices = td.get("p", [])
                pep_first_ask = td.get("fa")
            except Exception:
                pass

        if OsmiumParams.SYMBOL in state.order_depths:
            result[OsmiumParams.SYMBOL] = self._trade_osmium(state, prices)

        if PepperParams.SYMBOL in state.order_depths:
            orders, pep_first_ask = self._trade_pepper(state, pep_first_ask)
            result[PepperParams.SYMBOL] = orders

        return result, 0, json.dumps({"p": prices, "fa": pep_first_ask})

    def _trade_osmium(self, state: TradingState, prices: List[float]) -> List[Order]:
        P = OsmiumParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        best_bid = max(od.buy_orders.keys()) if od.buy_orders else None
        best_ask = min(od.sell_orders.keys()) if od.sell_orders else None

        if best_bid is not None and best_ask is not None:
            plain = (best_bid + best_ask) / 2
            bid_vol = od.buy_orders[best_bid]
            ask_vol = -od.sell_orders[best_ask]
            total = bid_vol + ask_vol
            if total > 0:
                micro = (best_bid * ask_vol + best_ask * bid_vol) / total
                mid = plain * (1 - P.MICROPRICE_WEIGHT) + micro * P.MICROPRICE_WEIGHT
            else:
                mid = plain
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

        # Urgency: widen take threshold when pos is far from target. Threshold
        # gated to avoid bleeding when inventory is near target (normal MM).
        raw_urg = (target_pos - pos) / P.POS_LIMIT
        mag = abs(raw_urg)
        if mag > P.URGENCY_THRESHOLD:
            ramp = (mag - P.URGENCY_THRESHOLD) / max(1e-9, 1.0 - P.URGENCY_THRESHOLD)
        else:
            ramp = 0.0
        urg_buy = max(0.0, ramp) if raw_urg > 0 else 0.0
        urg_sell = max(0.0, ramp) if raw_urg < 0 else 0.0

        spread = (best_ask - best_bid) if (best_bid is not None and best_ask is not None) else 99
        buy_edge = round(P.URGENCY_EDGE * urg_buy)
        sell_edge = round(P.URGENCY_EDGE * urg_sell)
        if spread <= P.NARROW_SPREAD and od.buy_orders and od.sell_orders:
            bv = sum(od.buy_orders.values())
            av = sum(-v for v in od.sell_orders.values())
            if bv > av:
                buy_edge += P.NARROW_EDGE
            elif av > bv:
                sell_edge += P.NARROW_EDGE

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

    def _trade_pepper(self, state: TradingState, first_ask):
        P = PepperParams
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if first_ask is None and od.sell_orders:
            first_ask = min(od.sell_orders.keys())

        if first_ask is None:
            return orders, first_ask

        cap = first_ask + P.BUY_MARGIN
        buy_cap = P.POS_LIMIT - pos
        if buy_cap <= 0:
            return orders, first_ask

        if od.sell_orders:
            for price in sorted(od.sell_orders.keys()):
                if price > cap or buy_cap <= 0:
                    break
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                buy_cap -= qty

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, cap, buy_cap))

        return orders, first_ask
