"""Advanced market-making: Avellaneda-Stoikov quoting with GARCH, Kalman, Hurst, OFI."""

from datamodel import OrderDepth, TradingState, Order
from typing import List
import json
import math


# =====================================================================
# Parameters
# =====================================================================

POS_LIMIT = 80

PEPPER = "INTARIAN_PEPPER_ROOT"
OSMIUM = "ASH_COATED_OSMIUM"

# --- Pepper fair value ---
PEP_MID_W    = 0.185
PEP_EMA_W    = 0.815
PEP_EMA_SPAN = 42          # EMA_42
PEP_TIME_SLOPE     = 0.00096
PEP_TIME_INTERCEPT = 88
PEP_GAUSS_AMP    = 72
PEP_GAUSS_DECAY  = 0.00012
PEP_GAUSS_CENTER = 28000
PEP_IMB_COEFF    = 0.42
PEP_REGIME_COEFF = 0.31
PEP_REGIME_TANH  = 0.0008
PEP_REGIME_CENTER = 52000

# --- Osmium fair value ---
OSM_MICRO_W  = 0.48
OSM_KALMAN_W = 0.32
OSM_SMA_W    = 0.20
OSM_SMA_WIN  = 14
OSM_HURST_K  = 0.05        # mean-reversion speed K_t
OSM_OFI_LAM  = 0.1         # lambda for OFI integral
OSM_OFI_K    = 0.01        # decay rate k

# --- Skew ---
BETA_SKEW       = 0.05
SKEW_TANH_SCALE = 0.00045
SKEW_SINE_COEFF = 0.31
SKEW_SINE_PERIOD = 72000

# --- Avellaneda-Stoikov quoting ---
GAMMA   = 0.5              # risk aversion gamma
DELTA   = 0.1              # volatility scaling Delta
K_DEPTH = 0.08             # market depth k
IMB_Q   = 0.5              # imbalance coefficient in quote
SKEW_Q  = 0.22             # skew coefficient in quote
HALF_SPREAD_MIN = 3        # minimum half-spread to ensure profitability

# --- Fair_final blend ---
FAIR_PEP_W   = 0.41
FAIR_OSM_W   = 0.33
FAIR_FLOW_W  = 0.18
FAIR_HAWK_W  = 0.08

# --- GARCH(1,1) ---
GARCH_OMEGA = 0.5
GARCH_ALPHA = 0.1
GARCH_BETA  = 0.85

# --- Kalman ---
KALMAN_Q = 0.1              # process noise
KALMAN_R = 1.0              # measurement noise


# =====================================================================
# Trader
# =====================================================================

class Trader:

    def run(self, state: TradingState):
        result = {}

        # ---- Load persisted state ----
        td = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                pass

        pep_ema      = td.get("pe", None)
        pep_gvar     = td.get("pg", 1.0)
        pep_last_mid = td.get("pl", None)
        pep_regime   = td.get("pr", 0.0)

        osm_prices   = td.get("op", [])
        osm_kx       = td.get("okx", None)
        osm_kp       = td.get("okp", 1.0)
        osm_ofi      = td.get("oo", 0.0)
        osm_havg     = td.get("oh", None)

        hawkes_lam   = td.get("hl", 0.1)   # Hawkes intensity

        t = state.timestamp

        # Will store per-product fair values for Fair_final
        f_pepper_val = None
        f_osmium_val = None
        flow_signal  = 0.0

        # =============================================================
        # PEPPER
        # =============================================================
        if PEPPER in state.order_depths:
            od  = state.order_depths[PEPPER]
            pos = state.position.get(PEPPER, 0)

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = float(best_bid)
            elif best_ask is not None:
                mid = float(best_ask)
            else:
                mid = pep_last_mid if pep_last_mid is not None else 12000.0

            # EMA_42
            alpha = 2.0 / (PEP_EMA_SPAN + 1)
            if pep_ema is None:
                pep_ema = mid
            else:
                pep_ema = alpha * mid + (1 - alpha) * pep_ema

            # GARCH(1,1) volatility
            if pep_last_mid is not None:
                ret = mid - pep_last_mid
                pep_gvar = GARCH_OMEGA + GARCH_ALPHA * ret * ret + GARCH_BETA * pep_gvar
            sigma = math.sqrt(max(pep_gvar, 0.01))

            # Book imbalance
            v_bid = sum(od.buy_orders.values()) if od.buy_orders else 0
            v_ask = sum(-v for v in od.sell_orders.values()) if od.sell_orders else 0
            imb = (v_bid - v_ask) / (v_bid + v_ask) if (v_bid + v_ask) > 0 else 0.0

            # Regime bias (smoothed trend indicator)
            if pep_last_mid is not None:
                if mid > pep_ema:
                    pep_regime = min(pep_regime + 0.1, 1.0)
                else:
                    pep_regime = max(pep_regime - 0.1, -1.0)

            # F_pepper(t)
            f_pepper_val = (
                PEP_MID_W * mid + PEP_EMA_W * pep_ema
                + PEP_TIME_SLOPE * t + PEP_TIME_INTERCEPT
                + PEP_GAUSS_AMP * math.exp(-PEP_GAUSS_DECAY * (t - PEP_GAUSS_CENTER) ** 2)
                + PEP_IMB_COEFF * imb * sigma
                + PEP_REGIME_COEFF * math.tanh(PEP_REGIME_TANH * (t - PEP_REGIME_CENTER)) * pep_regime
            )

            # Skew
            skew = (
                BETA_SKEW * pos * (1 + 0.62 * math.tanh(SKEW_TANH_SCALE * abs(pos)))
                + SKEW_SINE_COEFF * (POS_LIMIT - abs(pos)) / POS_LIMIT
                  * math.sin(2 * math.pi * t / SKEW_SINE_PERIOD)
            )

            # Avellaneda-Stoikov reservation price adjustment
            q = pos
            reserve_adj = DELTA * GAMMA * q + SKEW_Q * skew + IMB_Q * imb
            fair_adj = f_pepper_val - reserve_adj

            result[PEPPER] = self._make_orders(PEPPER, od, pos, fair_adj)

            pep_last_mid = mid
            flow_signal += imb * sigma  # contribution to Fair_final flow term

        # =============================================================
        # OSMIUM
        # =============================================================
        if OSMIUM in state.order_depths:
            od  = state.order_depths[OSMIUM]
            pos = state.position.get(OSMIUM, 0)

            best_bid = max(od.buy_orders) if od.buy_orders else None
            best_ask = min(od.sell_orders) if od.sell_orders else None

            if best_bid is not None and best_ask is not None:
                mid = (best_bid + best_ask) / 2
            elif best_bid is not None:
                mid = float(best_bid)
            elif best_ask is not None:
                mid = float(best_ask)
            else:
                mid = osm_kx if osm_kx is not None else 10000.0

            # Microprice
            if best_bid is not None and best_ask is not None:
                bv1 = od.buy_orders.get(best_bid, 0)
                av1 = -od.sell_orders.get(best_ask, 0)
                if bv1 + av1 > 0:
                    microprice = (best_bid * av1 + best_ask * bv1) / (bv1 + av1)
                else:
                    microprice = mid
            else:
                microprice = mid

            # SMA_14
            osm_prices.append(mid)
            if len(osm_prices) > OSM_SMA_WIN:
                osm_prices[:] = osm_prices[-OSM_SMA_WIN:]
            sma = sum(osm_prices) / len(osm_prices)

            # Kalman filter
            if osm_kx is None:
                osm_kx = mid
                osm_kp = 1.0
            else:
                osm_kp += KALMAN_Q
                k_gain = osm_kp / (osm_kp + KALMAN_R)
                osm_kx = osm_kx + k_gain * (mid - osm_kx)
                osm_kp = (1 - k_gain) * osm_kp

            # Hurst-based mean reversion: K_t * (m_t - H_S(t-1))
            if osm_havg is None:
                osm_havg = mid
            else:
                osm_havg = 0.95 * osm_havg + 0.05 * mid
            hurst_term = OSM_HURST_K * (mid - osm_havg)

            # Exponentially-weighted OFI integral
            v_bid = sum(od.buy_orders.values()) if od.buy_orders else 0
            v_ask = sum(-v for v in od.sell_orders.values()) if od.sell_orders else 0
            ofi = (v_bid - v_ask) / (v_bid + v_ask) if (v_bid + v_ask) > 0 else 0.0
            osm_ofi = osm_ofi * math.exp(-OSM_OFI_K) + OSM_OFI_LAM * ofi

            # F_osmium(t)
            f_osmium_val = (
                OSM_MICRO_W * microprice
                + OSM_KALMAN_W * osm_kx
                + OSM_SMA_W * sma
                + hurst_term
                + osm_ofi
            )

            # Skew
            skew = (
                BETA_SKEW * pos * (1 + 0.62 * math.tanh(SKEW_TANH_SCALE * abs(pos)))
                + SKEW_SINE_COEFF * (POS_LIMIT - abs(pos)) / POS_LIMIT
                  * math.sin(2 * math.pi * t / SKEW_SINE_PERIOD)
            )

            # Avellaneda-Stoikov reservation price adjustment
            q = pos
            reserve_adj = DELTA * GAMMA * q + SKEW_Q * skew + IMB_Q * ofi
            fair_adj = f_osmium_val - reserve_adj

            result[OSMIUM] = self._make_orders(OSMIUM, od, pos, fair_adj)

            flow_signal += ofi  # contribution to Fair_final flow term

        # ----- Hawkes intensity update (self-exciting from trade arrivals) -----
        n_trades = 0
        for sym in [PEPPER, OSMIUM]:
            if sym in state.market_trades:
                n_trades += len(state.market_trades[sym])
        hawkes_lam = hawkes_lam * math.exp(-0.001) + 0.05 * n_trades

        # Fair_final (informational / could be used for cross-asset hedging)
        # Fair_final = 0.41*F_pep + 0.33*F_osm + 0.18*(dV/sqrt(dt)) + 0.08*Ei[hawkes]
        # We compute it but the per-product quoting already uses F_pepper / F_osmium
        # dt = 100 (tick interval)
        # Ei approximation: Ei(x) ~ ln(x) + 0.5772 for small x
        hawkes_integral = hawkes_lam * 5000  # integral of intensity over next 5000 ts
        if hawkes_integral > 0:
            ei_approx = math.log(hawkes_integral) + 0.5772
        else:
            ei_approx = 0.0

        sqrt_dt = math.sqrt(100)
        fair_final = (
            FAIR_PEP_W * (f_pepper_val if f_pepper_val is not None else 0)
            + FAIR_OSM_W * (f_osmium_val if f_osmium_val is not None else 0)
            + FAIR_FLOW_W * (flow_signal / sqrt_dt)
            + FAIR_HAWK_W * ei_approx
        )
        # fair_final stored but not directly used for quoting (each product uses its own F)

        # ---- Save state ----
        new_td = json.dumps({
            "pe": pep_ema, "pg": pep_gvar, "pl": pep_last_mid, "pr": pep_regime,
            "op": osm_prices, "okx": osm_kx, "okp": osm_kp, "oo": osm_ofi, "oh": osm_havg,
            "hl": hawkes_lam,
        })

        return result, 0, new_td

    # ------------------------------------------------------------------
    # Order generation: take mispriced liquidity, then post inside walls
    # ------------------------------------------------------------------
    def _make_orders(
        self, symbol: str, od: OrderDepth, pos: int, fair: float,
    ) -> List[Order]:
        orders: List[Order] = []
        buy_cap  = POS_LIMIT - pos
        sell_cap = POS_LIMIT + pos

        best_bid = max(od.buy_orders) if od.buy_orders else None
        best_ask = min(od.sell_orders) if od.sell_orders else None
        fv = round(fair)

        # Phase 1 — aggressively take mispriced orders
        if od.sell_orders:
            for price in sorted(od.sell_orders):
                if price < fair and buy_cap > 0:
                    qty = min(-od.sell_orders[price], buy_cap)
                    orders.append(Order(symbol, price, qty))
                    buy_cap -= qty

        if od.buy_orders:
            for price in sorted(od.buy_orders, reverse=True):
                if price > fair and sell_cap > 0:
                    qty = min(od.buy_orders[price], sell_cap)
                    orders.append(Order(symbol, price, -qty))
                    sell_cap -= qty

        # Phase 2 — post just inside bot walls
        if best_bid is not None and best_ask is not None:
            our_bid = best_bid + 1
            our_ask = best_ask - 1
            if our_bid >= our_ask:
                our_bid = best_bid
                our_ask = best_ask
        elif best_bid is not None:
            our_bid = best_bid + 1
            our_ask = fv + HALF_SPREAD_MIN
        elif best_ask is not None:
            our_bid = fv - HALF_SPREAD_MIN
            our_ask = best_ask - 1
        else:
            our_bid = fv - HALF_SPREAD_MIN
            our_ask = fv + HALF_SPREAD_MIN

        if buy_cap > 0:
            orders.append(Order(symbol, our_bid, buy_cap))
        if sell_cap > 0:
            orders.append(Order(symbol, our_ask, -sell_cap))

        return orders
