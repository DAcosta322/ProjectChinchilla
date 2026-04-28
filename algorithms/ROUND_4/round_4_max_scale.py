"""Round 4 MAX-SCALE — uncap OTM voucher position without weakening ATM.

History:
  v1 (sub 504391, platform −$262):  spillover = SCALE_GAIN × vel_rel × POS_LIMIT
                                    Uncapped OTM (good +$1.5K) but weakened
                                    ATM (bad −$2K) by replacing the 3× DELTA
                                    formula. ATM saturated slower (vel_rel=0.8
                                    instead of 0.33).
  v2 (current):                     spillover = MAX(legacy 3×DELTA, SCALE_GAIN×1)
                                    Preserves ATM aggression (legacy wins for
                                    DELTA ≥ 0.4) AND uncaps OTM (new wins for
                                    DELTA < 0.4). LOOCV +$9.3K BT vs baseline.

Symptom that motivated the fix: sub 504391 trajectory was within $200 of sub
504109 the entire run. Per-product diff showed OTM gain +$1.5K canceled by
ATM loss −$2K (ATM positions never reached 300 because partial signals didn't
push them through the new lower saturation).
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Optional, Any
import json
import math


# ---------------------------------------------------------------------------
# Per-product params
# ---------------------------------------------------------------------------
class HydrogelParams:
    SYMBOL = "HYDROGEL_PACK"
    POS_LIMIT = 200
    INV_SKEW = 15
    ANCHOR_SPAN = 5000   # full-MR-day mean estimator (matches VEL).
    MR_STRENGTH = 5
    # Bayesian prior for EMA initialization. If set, the slow EMA anchor
    # starts at this value instead of the first observed price -> we get a
    # meaningful dev signal at t=0 instead of waiting thousands of ticks
    # for the EMA to drift toward the true cross-day mean. Memory: HYD
    # intraday mean is stable around 9990 across all observed days.
    FIXED_ANCHOR = 9990
    # Nonlinear MR boost: when |fv-anchor| > BOOST_THRESHOLD, add BOOST_GAIN
    # extra slope per unit excess. Steepens target curve at extremes so we
    # hit POS_LIMIT faster on big deviations. LOOCV-validated.
    BOOST_THRESHOLD = 20.0
    BOOST_GAIN = 3.0
    # Active flatten near mean: when |fv-anchor| < NEUTRAL_BAND, pull pos
    # toward 0 with strength NEUTRAL_GAIN. Frees up capacity for next move.
    # Sweep + LOOCV: HYD b=10 g=0.05 (broad plateau across band 5-20 at g=0.05).
    # Higher gains (>=0.5) cost spread on too many crossings and lose money;
    # 0.05 is the "gentle nudge toward flat" sweet spot.
    NEUTRAL_BAND = 10.0
    NEUTRAL_GAIN = 0.05
    # Trend-age detector. Tracks consecutive ticks where smoothed price velocity
    # exceeds TREND_THRESH with consistent sign. After MIN_AGE ticks (noise filter),
    # blends MR target toward trend-follow; ramps to full strength at FULL_AGE.
    # When velocity dies down, age decays and target reverts to MR (which now
    # says "buy" because dev still negative at trough = "buy them later").
    # Default off: TREND_STRENGTH=0.
    # FAST_SPAN remains 0 here so the MTS anchor-blend stays disabled (it
    # conflicts with FIXED_ANCHOR). The trend detector uses its own EMA
    # via TREND_FAST_SPAN below.
    # Trend-age detector (LOOCV-validated): velocity from per-tick change of
    # TREND_FAST_SPAN EMA, smoothed by TREND_VEL_SPAN. After MIN_AGE ticks of
    # consistent-sign velocity above THRESH, blend MR target toward trend-follow
    # (full strength at FULL_AGE). Decays back when velocity dies -> reverts to
    # MR which now says BUY at trough.
    TREND_FAST_SPAN = 200
    TREND_VEL_SPAN = 50
    TREND_THRESH = 0.005
    TREND_MIN_AGE = 50
    TREND_FULL_AGE = 2000
    TREND_STRENGTH = 0.3
    # Profit decay: scale target down once running in profit > PROFIT_DIST.
    PROFIT_DIST = 40
    PROFIT_RANGE = 40
    # Vol-adaptive MR strength (HYD only; VEL leaves VOL_SPAN=0). Clipped
    # to [0.5, 2.0] to bound effect.
    VOL_SPAN = 200
    VOL_REF = 2.0
    OFI_GAIN = 1.0
    OFI_SPAN = 20
    # Baseline passive quote size on the "wrong" side. Full POS_LIMIT
    # capacity reserved for the side target wants to load.
    BASE_QTY = 10
    # Aggressive-cross cap: limit Phase-1 crossings to (target - pos) +
    # AGG_OVERSHOOT. Without this, on tight-spread strikes we'd over-fill
    # past target via opportunistic crossings. LOOCV sweep: 10 wins held-out
    # D3, broad plateau 0-10. Disable with very large value.
    AGG_OVERSHOOT = 10
    EOD_START_FRAC = 0.7
    EOD_GAIN = 0.0
    EOD_FAST_SPAN = 500
    # Rolling-window drift bias: target += DRIFT_GAIN × (fv − rolling_EMA).
    # Adds a fast trend-follow on top of slow MR around fixed anchor.
    # Different from MR (which fights drift). DRIFT_GAIN=0 disables.
    DRIFT_SPAN = 1000
    DRIFT_GAIN = 0.0


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    # Re-tuned MR/BOOST after enabling vol-adaptive (VOL_SPAN=200, VOL_REF=0.7).
    # 48-config sweep: MR=12 BOOST_GAIN=5 BOOST_THRESHOLD=7 wins +$6.2K BT,
    # per-day positive on all 3 days (D1 +4.5K, D2 +1.9K, D3 -0.25K).
    MR_STRENGTH = 12
    # Cross-day mean for VEL is ~5250 (D0 5246, D1 5248, D2 5255).
    # Live day 486008 started at 5295, drifted to 5201, ended 5232 — exactly
    # the kind of day where a prior-based EMA init beats observed-price init.
    FIXED_ANCHOR = 5250
    # Nonlinear MR boost: VEL has tighter intraday std (~14), so smaller
    # threshold (5) and bigger gain (5) — small deviations get extra slope.
    BOOST_THRESHOLD = 7.0
    BOOST_GAIN = 5.0
    # Active flatten - VEL has tighter |dev| range so smaller band.
    NEUTRAL_BAND = 3.0
    NEUTRAL_GAIN = 0.05
    TREND_FAST_SPAN = 200
    TREND_VEL_SPAN = 50
    # Sweep + LOOCV: VEL th=0.02 S=0.6 MIN=200 FULL=2000 wins.
    # Higher threshold + bigger strength than HYD because VEL has lower vol -
    # smaller velocity values, so threshold needs to be tuned for VEL's scale.
    TREND_THRESH = 0.02
    TREND_MIN_AGE = 200
    TREND_FULL_AGE = 2000
    TREND_STRENGTH = 0.6
    PROFIT_DIST = 15
    PROFIT_RANGE = 15
    # Vol-adaptive MR_STRENGTH on VEL. Was 0 (off). 144-config sweep picks
    # VOL_SPAN=200, VOL_REF=0.7 as winning point (+$9.4K vs disabled).
    # vol_factor = clip(VOL_REF / std_est, 0.5, 2.0); on a typical VEL std~1.0,
    # factor ≈ 0.7 (slightly damped MR), but on calm stretches factor → 2.0.
    VOL_SPAN = 200
    VOL_REF = 0.7
    OFI_GAIN = 1.0
    OFI_SPAN = 20
    BASE_QTY = 10
    AGG_OVERSHOOT = 10
    EOD_START_FRAC = 0.7
    EOD_GAIN = 0.0
    EOD_FAST_SPAN = 500
    DRIFT_SPAN = 1000
    DRIFT_GAIN = 0.0


# ---- Voucher trading: 8 strikes, all using adaptive TV + VEL spillover ----
#
# For each VEV_K (call option on VEL):
#   intrinsic = max(0, S - K)  where S = velvet_mid
#   observed_TV = vev_mid - intrinsic
#   tv_estimate = EWMA(observed_TV, span=TV_EWMA_SPAN)
#   fair = intrinsic + tv_estimate
#
# Target combines:
#   1. MR around fair: -OWN_MR * (vev_mid - fair)  -> capture local oscillation
#   2. VEL spillover:  SPILLOVER_GAIN * vel_target_rel * POS_LIMIT  -> directional
# Clipped to +- POS_LIMIT.
#
# TV warm-up: skip trading until MIN_SAMPLES observations gathered.
# TV clipping: TV bounded to typical range so single bad ticks can't poison.

class _VevBase:
    POS_LIMIT = 300
    MIN_SAMPLES = 100
    # Sweep on max_scale: AGG_OVERSHOOT=0 wins +$2K BT (vouchers only — HYD/VEL
    # value has no effect since their targets are usually near ±POS_LIMIT).
    # With max_scale's uncapped OTM, opportunistic Phase 1 crossings past
    # target tend to over-buy on the wrong side; tightening to 0 helps.
    AGG_OVERSHOOT = 0
    # Delta-regression fair-value model:
    #   fair = alpha_t + DELTA * velvet_mid
    #   alpha_t = EWMA(vev_mid - DELTA * velvet_mid, span=ALPHA_SPAN)
    # DELTA is empirically estimated from D0/D1/D2 (very stable cross-day).
    # alpha_t captures the day-specific residual and adapts within a day.
    # Joint sweep on D0/D1/D2 (D3 removed, doesn't represent MR regime):
    # best = GAIN=3.0, ALPHA_SPAN=5000, OWN_MR=0 -> $381K total.
    # GAIN=3.0 means voucher hits POS_LIMIT at vel_target_rel=0.33/DELTA -
    # i.e. we max-position vouchers at small VEL signals. OWN_MR=0 because
    # the residual alpha (vev_mid - DELTA*vel) doesn't have exploitable MR.
    ALPHA_SPAN = 5000
    # Voucher target = MAX(legacy_3.0_DELTA_formula, SCALE_GAIN × POS_LIMIT × vel_rel)
    # Preserves ATM aggression (legacy 3×D dominates for D≥0.4) and uncaps
    # OTM (SCALE_GAIN × 300 dominates for D<0.4 → reaches POS_LIMIT instead
    # of being DELTA-capped at e.g. 144 for VEV_5400).
    SPILLOVER_GAIN = 3.0
    # 8-config sweep + LOOCV (post sub 504391 platform debug):
    #   SCALE_GAIN=1.25 wins LOOCV total +$9,286 (D1 +6.5K / D2 +2.1K / D3 +0.7K).
    #   In-sample: $405,200 (+$9.8K vs event_aggressive $395,350).
    #   Replaces previous SCALE_GAIN=1.25 (no-MAX) which was platform-flat
    #   because it inadvertently weakened ATM aggression.
    SCALE_GAIN = 1.25
    OWN_MR = 0
    BASE_QTY = 15


# Empirical deltas + observed mode bid-ask spread on R4 D1-D3.
# AGG_CROSS_MARGIN tested 0..3 uniform and per-strike — every non-zero value
# loses substantially in BT (margin=0.5: -$31K, margin=1.0: -$66K).
# Aggressive crossing IS the directional-alpha engine; spread cost paid back
# many-fold by directional gains. Margins kept = 0 here.
class VEV4000Params(_VevBase):
    SYMBOL = "VEV_4000"; STRIKE = 4000; DELTA = 1.000
    INV_SKEW = 3; OWN_MR = 5.0  # LOOCV-unanimous across all 3 hold-outs
    AGG_CROSS_MARGIN = 0   # spread mode 21 — but BT prefers no margin


class VEV4500Params(_VevBase):
    SYMBOL = "VEV_4500"; STRIKE = 4500; DELTA = 0.999
    INV_SKEW = 3; OWN_MR = 5.0  # LOOCV-unanimous
    AGG_CROSS_MARGIN = 0   # spread mode 16


class VEV5000Params(_VevBase):
    SYMBOL = "VEV_5000"; STRIKE = 5000; DELTA = 0.930
    INV_SKEW = 2; OWN_MR = 1.0
    AGG_CROSS_MARGIN = 0   # spread mode 6


class VEV5100Params(_VevBase):
    SYMBOL = "VEV_5100"; STRIKE = 5100; DELTA = 0.827
    INV_SKEW = 2; OWN_MR = 1.0
    AGG_CROSS_MARGIN = 0


class VEV5200Params(_VevBase):
    SYMBOL = "VEV_5200"; STRIKE = 5200; DELTA = 0.615
    INV_SKEW = 2; OWN_MR = 1.0
    AGG_CROSS_MARGIN = 0


class VEV5300Params(_VevBase):
    SYMBOL = "VEV_5300"; STRIKE = 5300; DELTA = 0.381
    INV_SKEW = 2; OWN_MR = 1.0
    AGG_CROSS_MARGIN = 0


class VEV5400Params(_VevBase):
    SYMBOL = "VEV_5400"; STRIKE = 5400; DELTA = 0.162
    INV_SKEW = 1; OWN_MR = 1.0
    AGG_CROSS_MARGIN = 0


class VEV5500Params(_VevBase):
    SYMBOL = "VEV_5500"; STRIKE = 5500; DELTA = 0.077
    INV_SKEW = 1; OWN_MR = 1.0
    AGG_CROSS_MARGIN = 0

VOUCHER_PARAMS = (VEV4000Params, VEV4500Params, VEV5000Params,
                  VEV5100Params, VEV5200Params, VEV5300Params,
                  VEV5400Params, VEV5500Params)


# ---------------------------------------------------------------------------
# Bot-event signal: VELVET only.
#
# Lead-lag analysis (bot_pattern_analysis.py) revealed:
#   Mark 67 buys VELVET → mid moves +2.0 within 1 tick, +1.6 at h=100,
#                         persists to +2.7 at h=2000.
#   Mark 49 sells VELVET → mirror -1.9 at h=1, -2.0 at h=100, -0.6 at h=2000.
# These are STRONG informed signals at short horizons.
#
# Earlier we tried "cumulative net flow" and it failed because both bots
# trade in monotone-fixed directions every day → the cumulative flow is
# structural (always growing toward +inf for 67 / -inf for 49) and has no
# day-relative information. This rewrite uses an EVENT-BASED EWMA: each
# new print fires a transient bump that decays away in ~50 ticks, so the
# signal value at any tick reflects only RECENT activity.
#
# Voucher (Mark 01/22) signals dropped — lead-lag was ~0 across horizons.
# Vouchers track VELVET delta; flow doesn't move them.
# ---------------------------------------------------------------------------
class BotEventParams:
    # 270-config sweep + LOOCV on R4 D1/D2/D3 (post-BT-lookahead-fix).
    # Best in-sample: SIGNAL_FULL=100, THRESH=10, HL=100, AT_MID=1.0, M14=0.1
    #   D1 +2,010 / D2 +6,760 / D3 +1,790 (total +10,560 vs baseline 384,790)
    # LOOCV held-out total: +3,380 (D1 +2,010, D2 −420, D3 +1,790).
    # D1 and D3 hold-outs independently pick this same config; D2 hold-out
    # picks a different config and loses 420 — so D2 is the weakest point.
    HALF_LIFE = 100
    # Signal magnitude (qty units) at which conviction = 100% → ±POS_LIMIT.
    # Larger SIGNAL_FULL = need more sustained activity to hit limit.
    SIGNAL_FULL = 100.0
    # Minimum |signal| to engage. Below this, plain MR target.
    SIGNAL_THRESH = 10.0
    # Weight on at-mid Mark 67/49 prints. 1.0 = full weight (LOOCV winner;
    # surprising given lead-lag was stronger on aggressive prints, but with
    # HL=100 the at-mid prints accumulate enough to add signal).
    AT_MID_WEIGHT = 1.0
    # Mark 14 VELVET — INVERSE signal (sub 502867 platform validation).
    # Lead-lag at h=50: Mark 14 BUY VEL → mid drops -0.77/-0.72/-0.56 across
    # D1/D2/D3. Code (in _ingest_velvet_event_aggressive) FADES Mark 14:
    # their BUY → -qty contribution, SELL → +qty.
    # 502867 BT showed only +$263 from this but platform delivered +$2,667
    # (10× translation, opposite of usual BT inflation). Trust platform.
    M14_WEIGHT = 1.0
    # Combine: 'blend' = conviction-weighted (signal weight = |conv|), preserves
    # MR target when signal is weak. 'max_mag' / 'override' less robust.
    COMBINE_MODE = "blend"
    # Conviction amplifier — multiplies |conv| before clipping to 1 in blend.
    # 256-config sweep result: AMP=1 is the OPTIMUM. Pushing bot dominance via
    # AMP>1 monotonically hurts (AMP=2 → +4.4K, AMP=4 → -12K, AMP=8 → -36K,
    # override mode → -44K). MR is the load-bearing alpha; bot signal is
    # additive (+10K on top), not a replacement. Don't re-explore.
    CONVICTION_AMP = 1.0


def _ingest_velvet_event_aggressive(market_trades, order_depths,
                                    velvet_event_box: List[float]) -> None:
    """Sharpened ingest: classify each VELVET trade as aggressive or at-mid
    using the current order_depths mid as reference, weight aggressive prints
    fully and at-mid prints by AT_MID_WEIGHT. Optionally fold in Mark 14."""
    if not market_trades:
        return
    velvet_trades = market_trades.get("VELVETFRUIT_EXTRACT")
    if not velvet_trades:
        return
    od = order_depths.get("VELVETFRUIT_EXTRACT")
    mid_now = _micro_mid(od) if od is not None else None
    for t in velvet_trades:
        buyer = getattr(t, "buyer", None)
        seller = getattr(t, "seller", None)
        qty = getattr(t, "quantity", 0)
        price = float(getattr(t, "price", 0))
        # Classify aggressiveness: aggressive_buy if price > mid (buyer
        # lifted ask); aggressive_sell if price < mid; at_mid otherwise.
        if mid_now is not None and price > mid_now + 0.5:
            agg_side = "buy"
            wt = 1.0
        elif mid_now is not None and price < mid_now - 0.5:
            agg_side = "sell"
            wt = 1.0
        else:
            agg_side = None  # at-mid: assign by counterparty role below
            wt = BotEventParams.AT_MID_WEIGHT
        # Mark 67 / Mark 49 contribution
        if buyer == "Mark 67" or seller == "Mark 49":
            # Both are "long-bullish" events; aggressive lifts especially so.
            sign = 1.0 if (agg_side != "sell") else -1.0  # rare: 67 sell = -
            if buyer == "Mark 67":
                velvet_event_box[0] += sign * qty * wt
            if seller == "Mark 49":
                velvet_event_box[0] += sign * qty * wt
        if seller == "Mark 67":
            velvet_event_box[0] -= qty * wt
        if buyer == "Mark 49":
            velvet_event_box[0] -= qty * wt
        # Mark 14 INVERSE on VELVET. Empirical lead-lag (sub 502867 finding):
        # Mark 14 BUY VEL → mid drops −0.77 / −0.72 / −0.56 at h=50 across
        # D1/D2/D3. Their direction is reliably wrong, so FADE them:
        # Mark 14 BUY → −qty (bearish signal), SELL → +qty (bullish).
        m14_w = BotEventParams.M14_WEIGHT
        if m14_w != 0:
            if buyer == "Mark 14":
                velvet_event_box[0] -= qty * wt * m14_w
            elif seller == "Mark 14":
                velvet_event_box[0] += qty * wt * m14_w


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def _update_avg_entry(prev_avg: float, prev_pos: int, own_trades: list) -> float:
    cur_pos = prev_pos
    avg = prev_avg if prev_pos != 0 else 0.0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION":
            qty = t.quantity
        elif getattr(t, "seller", None) == "SUBMISSION":
            qty = -t.quantity
        else:
            continue
        new_pos = cur_pos + qty
        if new_pos == 0:
            avg = 0.0
        elif cur_pos == 0 or (cur_pos > 0) != (new_pos > 0):
            avg = float(t.price)
        elif (cur_pos > 0) == (qty > 0):
            avg = (avg * cur_pos + t.price * qty) / new_pos
        cur_pos = new_pos
    return avg


def _profit_decay(P, pos: int, fv: float, avg_entry: float) -> float:
    if avg_entry <= 0 or pos == 0 or P.PROFIT_DIST <= 0:
        return 1.0
    sign = 1.0 if pos > 0 else -1.0
    profit_per_unit = (fv - avg_entry) * sign
    if profit_per_unit <= P.PROFIT_DIST:
        return 1.0
    excess = profit_per_unit - P.PROFIT_DIST
    return max(0.0, 1.0 - excess / max(1.0, P.PROFIT_RANGE))


def _book_top(od: OrderDepth):
    if not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, od.buy_orders[bb], ba, -od.sell_orders[ba]


def _ofi_step(prev, cur):
    if prev is None or cur is None:
        return 0.0
    bb1, bv1, ba1, av1 = prev
    bb2, bv2, ba2, av2 = cur
    if bb2 > bb1:
        e_bid = bv2
    elif bb2 < bb1:
        e_bid = -bv1
    else:
        e_bid = bv2 - bv1
    if ba2 > ba1:
        e_ask = av1
    elif ba2 < ba1:
        e_ask = -av2
    else:
        e_ask = av1 - av2
    return float(e_bid + e_ask)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:

    def bid(self):
        return 3337

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        td: Dict[str, Any] = {}
        if state.traderData:
            try:
                td = json.loads(state.traderData)
            except Exception:
                td = {}
        ema = td.get("ema", {})
        avg_entry = td.get("avg", {})
        prev_fv = td.get("pfv", {})
        var_ewma = td.get("var", {})
        prev_book = td.get("pbook", {})
        ofi_ewma = td.get("ofi", {})
        trend = td.get("trend", {})
        fast_ema = td.get("fema", {})
        trend_age = td.get("tage", {})
        trend_sign = td.get("tsign", {})
        # Bot-event signal (VELVET only). Per-tick decay then ingest new prints.
        velvet_event_box = [float(td.get("vev", 0.0))]
        if BotEventParams.HALF_LIFE > 0:
            decay = 0.5 ** (1.0 / BotEventParams.HALF_LIFE)
            velvet_event_box[0] *= decay
        _ingest_velvet_event_aggressive(state.market_trades,
                                        state.order_depths, velvet_event_box)

        # Refresh avg_entry from own_trades (only HYD/VEL use profit_decay)
        for P in (HydrogelParams, VelvetParams):
            if P.PROFIT_DIST <= 0:
                continue
            sym = P.SYMBOL
            own = state.own_trades.get(sym, []) if state.own_trades else []
            cur_pos = state.position.get(sym, 0)
            delta = sum(t.quantity if getattr(t, "buyer", None) == "SUBMISSION"
                       else -t.quantity if getattr(t, "seller", None) == "SUBMISSION" else 0
                       for t in own)
            avg_entry[sym] = _update_avg_entry(
                avg_entry.get(sym, 0.0), cur_pos - delta, own)

        # HYDROGEL
        if HydrogelParams.SYMBOL in state.order_depths:
            od = state.order_depths[HydrogelParams.SYMBOL]
            fv = _micro_mid(od)
            if fv is not None:
                _, orders = self._mr(state, HydrogelParams, fv, ema,
                                     avg_entry, prev_fv, var_ewma,
                                     prev_book, ofi_ewma, trend,
                                     fast_ema, trend_age, trend_sign)
                result[HydrogelParams.SYMBOL] = orders

        # VELVET
        velvet_mid: Optional[float] = None
        velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema,
                                      avg_entry, prev_fv, var_ewma,
                                      prev_book, ofi_ewma, trend,
                                      fast_ema, trend_age, trend_sign)
                # Bot-event AGGRESSIVE: when |signal| above SIGNAL_THRESH,
                # convert to ±POS_LIMIT target via piecewise-linear conviction:
                #   conviction = clip(signal / SIGNAL_FULL, -1, +1)
                #   sig_target = round(conviction * POS_LIMIT)
                # Combine with MR target by max-conviction (whichever has
                # greater |target| in the direction of the signal). Re-emit
                # orders with the new target.
                signal_val = velvet_event_box[0]
                if abs(signal_val) >= BotEventParams.SIGNAL_THRESH:
                    conv = signal_val / max(1e-6, BotEventParams.SIGNAL_FULL)
                    conv = max(-1.0, min(1.0, conv))
                    sig_target = int(round(conv * VelvetParams.POS_LIMIT))
                    mode = BotEventParams.COMBINE_MODE
                    if mode == "blend":
                        # Conviction-weighted: signal weight scales with |conv|.
                        # CONVICTION_AMP > 1 makes even small signals weigh heavily.
                        w = min(1.0, abs(conv) * BotEventParams.CONVICTION_AMP)
                        new_target = int(round(w * sig_target + (1 - w) * tp))
                    elif mode == "max_mag":
                        # Whichever target has bigger magnitude wins.
                        new_target = sig_target if abs(sig_target) > abs(tp) else tp
                    else:  # "override"
                        new_target = sig_target
                    new_target = max(-VelvetParams.POS_LIMIT,
                                     min(VelvetParams.POS_LIMIT, new_target))
                    if new_target != tp:
                        tp = new_target
                        orders = self._exec(state, VelvetParams, velvet_mid, tp)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        # Vouchers: pure mimic-VEL. target = velvet_target_rel * POS_LIMIT * DELTA.
        # When VEL goes long, voucher goes long proportionally to delta. fair set
        # at delta-regressed best estimate so quotes are sensible, but no own MR.
        if velvet_mid is not None:
            for P in VOUCHER_PARAMS:
                if P.SYMBOL not in state.order_depths:
                    continue
                od_v = state.order_depths[P.SYMBOL]
                vev_mid = _micro_mid(od_v)
                if vev_mid is None:
                    continue
                # Track residual alpha for fair-value pricing of quotes
                residual = vev_mid - P.DELTA * velvet_mid
                prev_alpha = ema.get(P.SYMBOL)
                a = 2.0 / (P.ALPHA_SPAN + 1.0)
                cur_alpha = residual if prev_alpha is None else (
                    a * residual + (1 - a) * prev_alpha)
                ema[P.SYMBOL] = cur_alpha
                fair = cur_alpha + P.DELTA * velvet_mid
                # MAX-SCALE: take MAX(legacy DELTA-formula, new uncap formula)
                # so ATM strikes keep original saturation (vel_rel=0.33→limit)
                # while OTM strikes get uncapped (reach POS_LIMIT instead of
                # being DELTA-capped at e.g. 144 for VEV_5400).
                # legacy = SPILLOVER_GAIN × vel_rel × POS_LIMIT × DELTA  (ATM-friendly)
                # new    = SCALE_GAIN    × vel_rel × POS_LIMIT          (OTM-friendly)
                # Both share sign(vel_rel); pick the one with greater magnitude.
                spillover_gain = getattr(P, "SPILLOVER_GAIN", 3.0)
                scale_gain = getattr(P, "SCALE_GAIN", 1.5)
                legacy = spillover_gain * velvet_target_rel * P.POS_LIMIT * P.DELTA
                new = scale_gain * velvet_target_rel * P.POS_LIMIT
                # max in absolute value, preserving sign
                spillover = legacy if abs(legacy) >= abs(new) else new
                own_mr = -getattr(P, "OWN_MR", 0) * (vev_mid - fair)
                target = int(round(spillover + own_mr))
                target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                result[P.SYMBOL] = self._exec(state, P, fair, target)

        return result, 0, json.dumps({
            "ema": ema, "avg": avg_entry,
            "pfv": prev_fv, "var": var_ewma,
            "pbook": prev_book, "ofi": ofi_ewma,
            "trend": trend, "fema": fast_ema,
            "tage": trend_age, "tsign": trend_sign,
            "vev": velvet_event_box[0]})

    def _mr(self, state: TradingState, P, fv: float,
            ema: Dict[str, float],
            avg_entry: Dict[str, float],
            prev_fv: Dict[str, float],
            var_ewma: Dict[str, float],
            prev_book: Dict[str, Any],
            ofi_ewma: Dict[str, float],
            trend: Dict[str, float],
            fast_ema: Dict[str, float],
            trend_age: Dict[str, int],
            trend_sign: Dict[str, int]):
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return 0, []

        vol_factor = 1.0
        if P.VOL_SPAN > 0:
            if P.SYMBOL in prev_fv:
                ret = fv - prev_fv[P.SYMBOL]
                ret = max(-50.0, min(50.0, ret))
                prev_var = var_ewma.get(P.SYMBOL, P.VOL_REF * P.VOL_REF)
                a = 2.0 / (P.VOL_SPAN + 1.0)
                var_ewma[P.SYMBOL] = a * (ret * ret) + (1 - a) * prev_var
            prev_fv[P.SYMBOL] = fv
            std_est = math.sqrt(max(0.01, var_ewma.get(P.SYMBOL, 1.0)))
            vol_factor = max(0.5, min(2.0, P.VOL_REF / std_est))

        ofi_smooth = 0.0
        if getattr(P, 'OFI_GAIN', 0) > 0 and getattr(P, 'OFI_SPAN', 0) > 0:
            cur_book = _book_top(od)
            ofi_one = _ofi_step(prev_book.get(P.SYMBOL), cur_book)
            prev_book[P.SYMBOL] = cur_book
            a_ofi = 2.0 / (P.OFI_SPAN + 1.0)
            ofi_smooth = a_ofi * ofi_one + (1 - a_ofi) * ofi_ewma.get(P.SYMBOL, 0.0)
            ofi_ewma[P.SYMBOL] = ofi_smooth

        # Static depth imbalance signal: sum(bid_vol L1+L2+L3) vs sum(ask_vol).
        # Positive ratio → more bids than asks → likely up-pressure.
        # Different from OFI (per-tick change of best level only).
        depth_imb = 0.0
        if getattr(P, 'DEPTH_IMB_GAIN', 0) > 0:
            sum_bid = sum(od.buy_orders.values())
            sum_ask = -sum(od.sell_orders.values())  # sell volumes are negative
            tot = sum_bid + sum_ask
            if tot > 0:
                depth_imb = (sum_bid - sum_ask) / tot

        # Rolling-window drift bias: track a faster EMA of fv and bias target
        # toward direction of (fv − fast_EMA). Captures sustained drift over
        # ~1000-tick window without killing the slow-anchor MR.
        drift_signal = 0.0
        drift_gain = getattr(P, "DRIFT_GAIN", 0.0)
        if drift_gain != 0:
            drift_key = P.SYMBOL + "_DRIFT"
            drift_span = getattr(P, "DRIFT_SPAN", 1000)
            prev_drift = fast_ema.get(drift_key)
            a_drift = 2.0 / (drift_span + 1.0)
            cur_drift_ema = fv if prev_drift is None else (
                a_drift * fv + (1 - a_drift) * prev_drift)
            fast_ema[drift_key] = cur_drift_ema
            drift_signal = fv - cur_drift_ema

        target = 0
        if P.ANCHOR_SPAN > 0:
            prev_L = ema.get(P.SYMBOL)
            alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
            if prev_L is None:
                # Bayesian prior: init EMA at FIXED_ANCHOR (historical mean)
                # if available, else at current price. Lets us trade on
                # day-1 deviations without waiting for EMA to drift.
                fa = getattr(P, "FIXED_ANCHOR", None)
                L = float(fa) if fa is not None else fv
            else:
                L = alpha * fv + (1 - alpha) * prev_L
            ema[P.SYMBOL] = L
            anchor = L

            # End-of-day asymmetric MR: blend anchor toward recent fv after
            # EOD_START_FRAC of the day has elapsed. As the day approaches
            # close, anchor → fast EMA → MR target shrinks → algo doesn't
            # fight sustained drifts. EOD_GAIN=0 disables.
            eod_gain = getattr(P, "EOD_GAIN", 0.0)
            if eod_gain > 0:
                # Use timestamp from state. We can't access state here, so
                # store the fast EMA in fast_ema dict and use a tick counter.
                eod_key = P.SYMBOL + "_EOD"
                tick_key = P.SYMBOL + "_TICK"
                cur_tick = trend.get(tick_key, 0) + 1
                trend[tick_key] = cur_tick
                # Day length: 10000 ticks (R4 standard). Blending ramps from
                # EOD_START_FRAC to 1.0 of the day.
                day_len = 10000.0
                start = getattr(P, "EOD_START_FRAC", 0.7) * day_len
                if cur_tick > start:
                    # Update fast EMA on fv
                    eod_span = getattr(P, "EOD_FAST_SPAN", 500)
                    a_eod = 2.0 / (eod_span + 1.0)
                    prev_eod = fast_ema.get(eod_key)
                    eod_fast = fv if prev_eod is None else (a_eod * fv + (1 - a_eod) * prev_eod)
                    fast_ema[eod_key] = eod_fast
                    # Linear ramp 0..1 over the EOD window
                    eod_progress = min(1.0, (cur_tick - start) / max(1.0, day_len - start))
                    blend_w = eod_gain * eod_progress
                    anchor = (1 - blend_w) * L + blend_w * eod_fast

            mr_eff = P.MR_STRENGTH * vol_factor
            dev = fv - anchor
            # Nonlinear boost: when |dev| exceeds BOOST_THRESHOLD, add extra
            # MR pressure proportional to excess. Steepens the target curve
            # at extremes so we hit POS_LIMIT faster on rare big deviations.
            bt_th = getattr(P, "BOOST_THRESHOLD", 1e9)
            bg = getattr(P, "BOOST_GAIN", 0.0)
            boost = 0.0
            if bg > 0 and abs(dev) > bt_th:
                excess = abs(dev) - bt_th
                boost = -bg * (1.0 if dev > 0 else -1.0) * excess
            # Active-flatten in the neutral zone: when |dev| < NEUTRAL_BAND,
            # add a position-decay term that drives target toward -pos so we
            # unwind to flat. Frees up POS_LIMIT for the next genuine move.
            #   pull = -NEUTRAL_GAIN * pos * (1 - |dev|/NEUTRAL_BAND)
            n_band = getattr(P, "NEUTRAL_BAND", 0.0)
            n_gain = getattr(P, "NEUTRAL_GAIN", 0.0)
            neutral_pull = 0.0
            if n_band > 0 and n_gain > 0 and abs(dev) < n_band:
                neutral_pull = -n_gain * pos * (1.0 - abs(dev) / n_band)
            raw = (-mr_eff * dev
                   + boost
                   + neutral_pull
                   + getattr(P, 'OFI_GAIN', 0) * ofi_smooth
                   + getattr(P, 'DEPTH_IMB_GAIN', 0) * depth_imb
                   + drift_gain * drift_signal)
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))

            # Trend-age modulation: detect SUSTAINED price-velocity (not regime)
            # via the per-tick change of fast_ema. Track consecutive ticks where
            # |velocity| > TREND_THRESH with consistent sign. Once age exceeds
            # MIN_AGE (filtered out noise), blend MR target toward trend-follow
            # so we sell during sustained downtrends and buy during uptrends.
            # When the velocity dies/reverses, age decays and target reverts to
            # MR - which naturally re-buys at the bottom (dev negative -> long).
            tr_thresh = getattr(P, "TREND_THRESH", 0.0)
            tr_min_age = getattr(P, "TREND_MIN_AGE", 0)
            tr_full_age = getattr(P, "TREND_FULL_AGE", 1)
            tr_strength = getattr(P, "TREND_STRENGTH", 0.0)
            tr_vel_span = getattr(P, "TREND_VEL_SPAN", 50)
            tr_fast_span = getattr(P, "TREND_FAST_SPAN", 0)
            if tr_thresh > 0 and tr_strength > 0 and tr_fast_span > 0:
                # Compute our own fast EMA (independent of MTS anchor-blend).
                # Init at FIXED_ANCHOR if available, else current fv.
                key = P.SYMBOL + "_TFAST"
                prev_tf = fast_ema.get(key)
                a_tf = 2.0 / (tr_fast_span + 1.0)
                if prev_tf is None:
                    fa = getattr(P, "FIXED_ANCHOR", None)
                    prev_tf = float(fa) if fa is not None else fv
                Tf = a_tf * fv + (1 - a_tf) * prev_tf
                fast_ema[key] = Tf
                velocity_raw = Tf - prev_tf
                prev_vel = trend.get(P.SYMBOL + "_VEL", 0.0)
                bv = 2.0 / (tr_vel_span + 1.0)
                velocity = bv * velocity_raw + (1.0 - bv) * prev_vel
                trend[P.SYMBOL + "_VEL"] = velocity

                if velocity > tr_thresh:
                    cur_sign = 1
                elif velocity < -tr_thresh:
                    cur_sign = -1
                else:
                    cur_sign = 0
                prev_age = trend_age.get(P.SYMBOL, 0)
                prev_sign = trend_sign.get(P.SYMBOL, 0)
                if cur_sign == 0:
                    new_age = max(0, prev_age - 2)
                    new_sign = prev_sign if new_age > 0 else 0
                elif cur_sign == prev_sign:
                    new_age = prev_age + 1
                    new_sign = cur_sign
                else:
                    new_age = 1
                    new_sign = cur_sign
                trend_age[P.SYMBOL] = new_age
                trend_sign[P.SYMBOL] = new_sign

                if new_age > tr_min_age and new_sign != 0:
                    span_a = max(1, tr_full_age - tr_min_age)
                    maturity = max(0.0, min(1.0, (new_age - tr_min_age) / span_a))
                    follow_w = tr_strength * maturity
                    # Sign convention: velocity > 0 = uptrend, < 0 = downtrend.
                    # During uptrend (price rising), trend-follow says LONG.
                    # During downtrend, trend-follow says SHORT.
                    follow_target = new_sign * P.POS_LIMIT
                    target = int(round((1 - follow_w) * target + follow_w * follow_target))
                    target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))

        decay = _profit_decay(P, pos, fv, avg_entry.get(P.SYMBOL, 0.0))
        target = int(round(target * decay))

        return target, self._do_orders(P, fv, pos, target, od)

    def _exec(self, state: TradingState, P, fv: float, target: int) -> List[Order]:
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return []
        return self._do_orders(P, fv, pos, target, od)

    def _do_orders(self, P, fv: float, pos: int, target_pos: int,
                   od: OrderDepth) -> List[Order]:
        orders: List[Order] = []
        skew = P.INV_SKEW * (pos - target_pos) / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

        # Cap aggressive cross by target distance (with small overshoot).
        # Without this, on tight-spread low-delta vouchers (e.g. VEV_5400
        # target maxes at ±146 of POS_LIMIT 300), aggressive Phase-1
        # crossing accumulates pos all the way to ±300 even though the
        # signal only justifies ±146. The overshoot allowance keeps us
        # taking opportunistic +EV crossings.
        agg_over = getattr(P, "AGG_OVERSHOOT", 50)
        agg_buy_cap = max(0, min(buy_cap, (target_pos - pos) + agg_over))
        agg_sell_cap = max(0, min(sell_cap, (pos - target_pos) + agg_over))

        # Voucher spread is structurally fixed per strike (mode 21/16/6/4/3/2/1/1
        # for VEV_4000..VEV_5500 across 70%+ of ticks). Aggressive crossing
        # without a margin pays the full half-spread — substantial slippage
        # on wider strikes. AGG_CROSS_MARGIN requires fv_eff to exceed the
        # ask (or be below the bid) by at least this many ticks before lifting.
        agg_margin = getattr(P, "AGG_CROSS_MARGIN", 0.0)
        for price in sorted(od.sell_orders.keys()):
            if price < fv_eff - agg_margin and agg_buy_cap > 0:
                qty = min(-od.sell_orders[price], agg_buy_cap)
                orders.append(Order(P.SYMBOL, price, qty))
                agg_buy_cap -= qty
                buy_cap -= qty
        for price in sorted(od.buy_orders.keys(), reverse=True):
            if price > fv_eff + agg_margin and agg_sell_cap > 0:
                qty = min(od.buy_orders[price], agg_sell_cap)
                orders.append(Order(P.SYMBOL, price, -qty))
                agg_sell_cap -= qty
                sell_cap -= qty

        base_bid = best_bid + 1
        base_ask = best_ask - 1
        if base_bid >= base_ask:
            base_bid = best_bid
            base_ask = best_ask
        shift = int(round(-skew))
        our_bid = min(base_bid + shift, int(fv_eff))
        our_ask = max(base_ask + shift, int(fv_eff) + 1)
        if our_ask <= our_bid:
            our_ask = our_bid + 1

        # Size passive quotes by target deviation, not full position-limit
        # capacity. Prevents the bid from absorbing market sells when target
        # says shrink (and vice versa). BASE_QTY keeps a baseline both sides.
        # For low-delta vouchers (e.g. VEV_5400 DELTA=0.16), the spillover
        # target maxes at ~146 (well below POS_LIMIT 300), but the unscaled
        # BASE_QTY=15 still posts every tick and accumulates 150 lots past
        # target via passive fills on tight-spread strikes. Scaling base by
        # DELTA caps the over-fill in proportion to the strike's actual
        # signal magnitude.
        base = getattr(P, "BASE_QTY", 10)
        delta_scale = getattr(P, "DELTA", 1.0)
        eff_base = max(2, int(round(base * delta_scale)))
        bid_qty = min(buy_cap, max(0, target_pos - pos) + eff_base)
        ask_qty = min(sell_cap, max(0, pos - target_pos) + eff_base)

        if bid_qty > 0:
            orders.append(Order(P.SYMBOL, our_bid, bid_qty))
        if ask_qty > 0:
            orders.append(Order(P.SYMBOL, our_ask, -ask_qty))

        # Tier-2 passive quotes (multi-level book). Place a smaller quote at
        # the book's best price (joining the queue) to catch flow that hits
        # the touch level. T2_FRAC=0 disables. T2 quotes only consume what's
        # left of buy/sell capacity after the primary quote.
        t2_frac = getattr(P, "TIER2_FRAC", 0.0)
        if t2_frac > 0:
            t2_size = max(1, int(round(eff_base * t2_frac)))
            # Bid at best_bid (joining), only if our primary quote was inside spread
            if bid_qty > 0 and our_bid > best_bid:
                t2_bid_qty = min(buy_cap - bid_qty, t2_size)
                if t2_bid_qty > 0:
                    orders.append(Order(P.SYMBOL, best_bid, t2_bid_qty))
            # Ask at best_ask (joining)
            if ask_qty > 0 and our_ask < best_ask:
                t2_ask_qty = min(sell_cap - ask_qty, t2_size)
                if t2_ask_qty > 0:
                    orders.append(Order(P.SYMBOL, best_ask, -t2_ask_qty))
        return orders
