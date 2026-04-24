"""Round 3 - HYDROGEL_PACK + VELVETFRUIT_EXTRACT + VEV_4000 + VEV_4500.

Strategy rationale (validated against day 0/1/2 historical data,
3-day backtest PnL: +54,133):

- HYDROGEL_PACK: spread ~15, mean-reverts around 10000. Pure penny-best
  MM. Fair = instantaneous micro-mid. Inventory skew shifts both quotes
  toward fv_eff so a severely skewed side crosses the book to rebalance.
- VELVETFRUIT_EXTRACT: spread ~5, drifts up day-over-day. Strong
  inventory skew (INV_SKEW=8) auto-covers when persistent one-sided
  flow builds up position.
- VEV_4000, VEV_4500: deep ITM, TV ~= 0 (std < 1). Fair = S_mid - K
  from live underlying each tick - non-stale, so aggressive takes are
  safe. No inventory skew needed since fair is accurate.

Skipped vouchers:
- VEV_5000, VEV_5100, VEV_5200: <= 8 market trades / day.
  Phase 2 passive fills nearly absent; aggressive takes bleed.
- VEV_5300, VEV_5400, VEV_5500: tighter spreads (1-2 ticks) and
  lagging time-value estimates over-trigger takes - net loser.
- VEV_6000, VEV_6500: pinned at 0.5 floor all day - no edge.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional


class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 2


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 8


class VEV4000Params:
    SYMBOL = "VEV_4000"
    STRIKE = 4000
    POS_LIMIT = 300
    INV_SKEW = 0


class VEV4500Params:
    SYMBOL = "VEV_4500"
    STRIKE = 4500
    POS_LIMIT = 300
    INV_SKEW = 0


def _micro_mid(od: OrderDepth) -> Optional[float]:
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    bv = od.buy_orders[bb]
    av = -od.sell_orders[ba]
    if bv + av <= 0:
        return (bb + ba) / 2.0
    return (bb * av + ba * bv) / (bv + av)


class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                result[HydrogelParams.SYMBOL] = self._mm_with_fair(
                    state, HydrogelParams, fv)

        velvet_mid: Optional[float] = None
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                result[VelvetParams.SYMBOL] = self._mm_with_fair(
                    state, VelvetParams, velvet_mid)

        if velvet_mid is not None:
            for P in (VEV4000Params, VEV4500Params):
                if P.SYMBOL in state.order_depths:
                    fv = velvet_mid - P.STRIKE
                    result[P.SYMBOL] = self._mm_with_fair(state, P, fv)

        return result, 0, ""

    # Unified MM: takes mispriced orders past fv_eff, then posts
    # penny-best quotes shifted by inventory so the skewed side
    # crosses the book when severely imbalanced (auto-rebalance).
    def _mm_with_fair(self, state: TradingState, P, fv: float
                      ) -> List[Order]:
        orders: List[Order] = []
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)

        if not od.buy_orders or not od.sell_orders:
            return orders

        skew = P.INV_SKEW * pos / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())

        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        # Aggressive takes strictly past fv_eff
        for price in sorted(od.sell_orders.keys()):
            if price < fv_eff and buy_cap > 0:
                qty = min(-od.sell_orders[price], buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fv_eff and sell_cap > 0:
                qty = min(od.buy_orders[price], sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty))
                sell_cap -= qty

        # Penny-best posts, then shifted by inventory skew. Short
        # inventory produces positive shift -> bid may cross best_ask
        # and auto-cover. Long is symmetric.
        base_bid = best_bid + 1
        base_ask = best_ask - 1
        if base_bid >= base_ask:
            base_bid = best_bid
            base_ask = best_ask

        shift = int(round(-skew))
        our_bid = base_bid + shift
        our_ask = base_ask + shift

        # Safety clamps: never post bid above own fair or ask below.
        our_bid = min(our_bid, int(fv_eff))
        our_ask = max(our_ask, int(fv_eff) + 1)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        if buy_cap > 0:
            orders.append(Order(P.SYMBOL, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(P.SYMBOL, our_ask, -sell_cap))

        return orders
