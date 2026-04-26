"""Round 3 voucher MR - mean-reversion on the delta-regression residual.

Sister of round_3_voucher.py. The default voucher.py uses pure VEL spillover
(target = SPILLOVER_GAIN * vel_target_rel * POS_LIMIT * DELTA), which keeps
voucher positions at extremes whenever VEL is at extremes - regardless of
whether the voucher is mispriced. During platform phases when residual is
near its mean (voucher fairly priced), spillover holds us in an inventory
risk that earns nothing.

This version replaces spillover with OWN_MR on the residual:
    residual_dev = vev_mid - fair  (where fair = alpha_t + DELTA*velvet_mid)
    target = -MR_GAIN * residual_dev   (clipped to +-POS_LIMIT)

A NEUTRAL_BAND deadband forces target = 0 when |residual_dev| <= NEUTRAL_BAND.
That is the "platform phase near mean" unwind: when the voucher is correctly
priced relative to VEL, we go flat instead of carrying inventory risk.

HYD/VEL unchanged from round_3_voucher.py.
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
    # Optional multi-timescale anchor blend (DISABLED by default).
    # FAST_SPAN > 0 enables: anchor = (1-w)*slow + w*fast, where
    # w = clip((|fast - slow| - DEADBAND) / SCALE, 0, 1). Useful only if
    # the live regime turns out to be persistent sub-day drift, which the
    # full-day MR assumption doesn't support.
    FAST_SPAN = 0
    DRIFT_DEADBAND = 3.0
    DRIFT_SCALE = 10.0
    PROFIT_DIST = 40
    PROFIT_RANGE = 40
    VOL_SPAN = 200
    VOL_REF = 2.0
    TOX_SPAN = 50
    TOX_FULL_OFF = 0
    # OFI_GAIN held at 1.0. A sweep showed 2.0 gave +$970 in-sample but LOOCV
    # (tune on 3 days, score on held-out) returned -$2.2K vs baseline; per-day
    # signs were mixed (D0 hurt by $2K, D1/D2/D3 helped); VEL OFI_GAIN sweep
    # showed VEL doesn't benefit either. Verdict: 2.0 was overfit noise.
    OFI_GAIN = 1.0
    OFI_SPAN = 20
    # Holt level+trend anchor: 0 = classical EMA. >0 enables Holt's method
    # with the given trend-EWMA span. During a sustained drift the anchor
    # catches up to fv much faster, so target doesn't pile long on the way
    # down. Sweep below to find safe value (off by default).
    HOLT_TREND_SPAN = 0
    # Range-position blend: target += RANGE_GAIN * (mid - fv) over rolling
    # [lo, hi] of last WINDOW fvs. Disabled by default (RANGE_GAIN=0) - sweep
    # showed even gain=0.5 costs ~$3.8K on historicals to gain ~$280 on D3.
    # Keep the wiring for regime-specific tuning; the historical loss
    # outweighs the platform-day stickiness benefit.
    RANGE_WINDOW = 300
    RANGE_MIN = 8
    RANGE_GAIN = 0
    # Baseline passive size on the "wrong" side (when target says shrink).
    # Preserves two-sided spread capture; full POS_LIMIT exposure is reserved
    # for the side that target wants.
    BASE_QTY = 10


class VelvetParams:
    SYMBOL = "VELVETFRUIT_EXTRACT"
    POS_LIMIT = 200
    INV_SKEW = 3
    ANCHOR_SPAN = 5000
    MR_STRENGTH = 10
    PROFIT_DIST = 15
    PROFIT_RANGE = 15
    VOL_SPAN = 0
    VOL_REF = 1.0
    TOX_SPAN = 50
    TOX_FULL_OFF = 0
    OFI_GAIN = 1.0
    OFI_SPAN = 20
    HOLT_TREND_SPAN = 0
    # Range disabled for VEL - long EMA already wins; range hurts the
    # day-scale MR by being too short-horizon.
    RANGE_WINDOW = 0
    RANGE_MIN = 0
    RANGE_GAIN = 0
    BASE_QTY = 10


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
    PROFIT_DIST = 0
    PROFIT_RANGE = 0
    VOL_SPAN = 0
    TOX_SPAN = 0; TOX_FULL_OFF = 0
    MIN_SAMPLES = 100
    # Delta-regression fair-value model:
    #   fair = alpha_t + DELTA * velvet_mid
    #   alpha_t = EWMA(vev_mid - DELTA * velvet_mid, span=ALPHA_SPAN)
    ALPHA_SPAN = 5000
    # Combined target = spillover + MR-with-band.
    #   spillover  = SPILLOVER_GAIN * velvet_target_rel * POS_LIMIT * DELTA
    #              -> the directional VEL alpha that drove voucher.py to $381K.
    #   mr_term    = -MR_GAIN * (vev_mid - fair)  outside +-NEUTRAL_BAND
    #              = 0  inside the band (so target collapses toward 0 when
    #                   the voucher is fairly priced AND VEL is near mean).
    # The band-driven unwind only matters when spillover is also small (i.e.
    # both VEL and the residual sit near their respective means). In those
    # platform phases we go neutral instead of carrying inventory risk.
    SPILLOVER_GAIN = 3.0   # matches voucher.py default
    MR_GAIN = 0            # per-strike below
    NEUTRAL_BAND = 0       # per-strike below
    BASE_QTY = 15


# Empirical DELTAs from D0/D1/D2 regression (stable cross-day, max delta change <0.1).
# Per-strike residual std (avg of D0-D2):
#   K=4000: 0.83  4500: 0.76  5000: 0.73  5100: 1.18  5200: 1.57
#   K=5300: 1.24  5400: 1.31  5500: 0.62
# MR_GAIN = POS_LIMIT(300) / (3*sigma) -> hit limit at 3-sigma residual deviation.
# NEUTRAL_BAND = 1*sigma -> exit during typical fluctuation phases.
# Tuned via grid sweep on D0/D1/D2 (SPILLOVER_GAIN x MR_GAIN x NEUTRAL_BAND).
# Best: SG=3.0, MR_GAIN=10, NEUTRAL_BAND=2.0  -> $380,710 BT
# (within $445 of voucher.py's pure-spillover $381,155). MR_GAIN > 10 always
# fights the dominant directional alpha; lower NEUTRAL_BAND lets MR fire on
# noise. The BT-detectable MR-on-residual alpha is small - this version is
# primarily an insurance against live-day platform phases where pure spillover
# might leave inventory drift unaddressed.
class VEV4000Params(_VevBase):
    SYMBOL = "VEV_4000"; STRIKE = 4000; DELTA = 1.000
    INV_SKEW = 3
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV4500Params(_VevBase):
    SYMBOL = "VEV_4500"; STRIKE = 4500; DELTA = 0.999
    INV_SKEW = 3
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV5000Params(_VevBase):
    SYMBOL = "VEV_5000"; STRIKE = 5000; DELTA = 0.930
    INV_SKEW = 2
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV5100Params(_VevBase):
    SYMBOL = "VEV_5100"; STRIKE = 5100; DELTA = 0.827
    INV_SKEW = 2
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV5200Params(_VevBase):
    SYMBOL = "VEV_5200"; STRIKE = 5200; DELTA = 0.615
    INV_SKEW = 2
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV5300Params(_VevBase):
    SYMBOL = "VEV_5300"; STRIKE = 5300; DELTA = 0.381
    INV_SKEW = 2
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV5400Params(_VevBase):
    SYMBOL = "VEV_5400"; STRIKE = 5400; DELTA = 0.162
    INV_SKEW = 1
    MR_GAIN = 10; NEUTRAL_BAND = 2.0


class VEV5500Params(_VevBase):
    SYMBOL = "VEV_5500"; STRIKE = 5500; DELTA = 0.077
    INV_SKEW = 1
    MR_GAIN = 10; NEUTRAL_BAND = 2.0

VOUCHER_PARAMS = (VEV4000Params, VEV4500Params, VEV5000Params,
                  VEV5100Params, VEV5200Params, VEV5300Params,
                  VEV5400Params, VEV5500Params)


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


def _update_toxicity(P, prev_tox: float, own_trades: list, fv_now: float) -> float:
    if not own_trades or P.TOX_SPAN <= 0:
        return prev_tox * 0.95
    bad = 0.0
    n = 0
    for t in own_trades:
        if getattr(t, "buyer", None) == "SUBMISSION":
            adv = max(0.0, t.price - fv_now)
        elif getattr(t, "seller", None) == "SUBMISSION":
            adv = max(0.0, fv_now - t.price)
        else:
            continue
        bad += adv
        n += 1
    if n == 0:
        return prev_tox * 0.95
    avg_adverse = bad / n
    a = 2.0 / (P.TOX_SPAN + 1.0)
    return a * avg_adverse + (1 - a) * prev_tox


def _range_component(P, fv: float, hist: List[float]) -> float:
    """Append fv to rolling history, return RANGE_GAIN*(mid-fv) once full."""
    win = getattr(P, "RANGE_WINDOW", 0)
    gain = getattr(P, "RANGE_GAIN", 0)
    if win <= 0 or gain <= 0:
        return 0.0
    hist.append(fv)
    if len(hist) > win:
        del hist[: len(hist) - win]
    if len(hist) < win:
        return 0.0
    lo = min(hist)
    hi = max(hist)
    rng = hi - lo
    if rng < getattr(P, "RANGE_MIN", 0):
        return 0.0
    midpoint = (hi + lo) / 2.0
    return gain * (midpoint - fv)


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
        tv = td.get("tv", {})
        tv_n = td.get("tvn", {})
        prev_fv = td.get("pfv", {})
        var_ewma = td.get("var", {})
        tox = td.get("tox", {})
        prev_book = td.get("pbook", {})
        ofi_ewma = td.get("ofi", {})
        hist = td.get("hist", {})
        trend = td.get("trend", {})
        fast_ema = td.get("fema", {})

        for P in (HydrogelParams, VelvetParams):
            if P.PROFIT_DIST <= 0 and P.TOX_FULL_OFF <= 0:
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
                own = state.own_trades.get(HydrogelParams.SYMBOL, []) if state.own_trades else []
                tox[HydrogelParams.SYMBOL] = _update_toxicity(
                    HydrogelParams, tox.get(HydrogelParams.SYMBOL, 0.0), own, fv)
                hist.setdefault(HydrogelParams.SYMBOL, [])
                _, orders = self._mr(state, HydrogelParams, fv, ema,
                                     avg_entry, prev_fv, var_ewma, tox,
                                     prev_book, ofi_ewma, hist, trend,
                                     fast_ema)
                result[HydrogelParams.SYMBOL] = orders

        # VELVET
        velvet_mid: Optional[float] = None
        velvet_target_rel = 0.0
        if VelvetParams.SYMBOL in state.order_depths:
            od = state.order_depths[VelvetParams.SYMBOL]
            velvet_mid = _micro_mid(od)
            if velvet_mid is not None:
                own = state.own_trades.get(VelvetParams.SYMBOL, []) if state.own_trades else []
                tox[VelvetParams.SYMBOL] = _update_toxicity(
                    VelvetParams, tox.get(VelvetParams.SYMBOL, 0.0), own, velvet_mid)
                hist.setdefault(VelvetParams.SYMBOL, [])
                tp, orders = self._mr(state, VelvetParams, velvet_mid, ema,
                                      avg_entry, prev_fv, var_ewma, tox,
                                      prev_book, ofi_ewma, hist, trend,
                                      fast_ema)
                velvet_target_rel = tp / VelvetParams.POS_LIMIT
                result[VelvetParams.SYMBOL] = orders

        # Vouchers: own-MR on the residual with NEUTRAL_BAND unwind. fair is
        # the delta-regressed estimate alpha_t + DELTA*velvet_mid. Inside the
        # band we go to 0 (active unwind during platform phases). Outside,
        # target = -MR_GAIN * (vev_mid - fair) clipped to +-POS_LIMIT.
        if velvet_mid is not None:
            for P in VOUCHER_PARAMS:
                if P.SYMBOL not in state.order_depths:
                    continue
                od_v = state.order_depths[P.SYMBOL]
                vev_mid = _micro_mid(od_v)
                if vev_mid is None:
                    continue
                residual = vev_mid - P.DELTA * velvet_mid
                prev_alpha = ema.get(P.SYMBOL)
                a = 2.0 / (P.ALPHA_SPAN + 1.0)
                cur_alpha = residual if prev_alpha is None else (
                    a * residual + (1 - a) * prev_alpha)
                ema[P.SYMBOL] = cur_alpha
                fair = cur_alpha + P.DELTA * velvet_mid
                resid_dev = vev_mid - fair
                if abs(resid_dev) <= P.NEUTRAL_BAND:
                    target = 0
                else:
                    sign = 1.0 if resid_dev > 0 else -1.0
                    excess = abs(resid_dev) - P.NEUTRAL_BAND
                    target = int(round(-sign * P.MR_GAIN * excess))
                # Optional spillover left wired but disabled by default
                if P.SPILLOVER_GAIN:
                    target += int(round(P.SPILLOVER_GAIN
                                        * velvet_target_rel
                                        * P.POS_LIMIT * P.DELTA))
                target = max(-P.POS_LIMIT, min(P.POS_LIMIT, target))
                result[P.SYMBOL] = self._exec(state, P, fair, target)

        return result, 0, json.dumps({
            "ema": ema, "avg": avg_entry, "tv": tv, "tvn": tv_n,
            "pfv": prev_fv, "var": var_ewma, "tox": tox,
            "pbook": prev_book, "ofi": ofi_ewma, "hist": hist,
            "trend": trend, "fema": fast_ema})

    def _mr(self, state: TradingState, P, fv: float,
            ema: Dict[str, float],
            avg_entry: Dict[str, float],
            prev_fv: Dict[str, float],
            var_ewma: Dict[str, float],
            tox: Dict[str, float],
            prev_book: Dict[str, Any],
            ofi_ewma: Dict[str, float],
            hist: Dict[str, List[float]],
            trend: Dict[str, float],
            fast_ema: Dict[str, float]):
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

        # Always advance the rolling history, even if RANGE inactive
        range_comp = _range_component(P, fv, hist.setdefault(P.SYMBOL, []))

        target = 0
        if P.ANCHOR_SPAN > 0:
            prev_L = ema.get(P.SYMBOL)
            holt_span = getattr(P, "HOLT_TREND_SPAN", 0)
            alpha = 2.0 / (P.ANCHOR_SPAN + 1.0)
            if prev_L is None:
                L = fv
                T = 0.0
            elif holt_span > 0:
                # Holt's level + trend: L tracks fv but is updated using
                # the trend-extrapolated previous estimate, so during a
                # sustained drift L catches up to fv much faster than a
                # plain EMA.
                prev_T = trend.get(P.SYMBOL, 0.0)
                beta = 2.0 / (holt_span + 1.0)
                L = alpha * fv + (1 - alpha) * (prev_L + prev_T)
                T = beta * (L - prev_L) + (1 - beta) * prev_T
                # Clip trend so a brief volatile period can't make L
                # diverge unboundedly.
                T = max(-2.0, min(2.0, T))
            else:
                L = alpha * fv + (1 - alpha) * prev_L
                T = 0.0
            ema[P.SYMBOL] = L
            trend[P.SYMBOL] = T
            slow_anchor = L

            # Multi-timescale ANCHOR BLEND with deadband: keep full slow
            # anchor on calm days (big fv-anchor -> big target). Only
            # blend toward fast anchor when drift exceeds DEADBAND, so
            # day-to-day oscillation noise doesn't sap the slow benefit.
            #   w = 0                                 if drift_dist <= DEADBAND
            #   w = clip((drift - DEADBAND) / SCALE)  otherwise
            fast_span = getattr(P, "FAST_SPAN", 0)
            if fast_span > 0:
                prev_fast = fast_ema.get(P.SYMBOL)
                a_fast = 2.0 / (fast_span + 1.0)
                F = fv if prev_fast is None else a_fast * fv + (1 - a_fast) * prev_fast
                fast_ema[P.SYMBOL] = F
                drift_dist = abs(F - slow_anchor)
                deadband = getattr(P, "DRIFT_DEADBAND", 0.0)
                scale = max(1e-6, getattr(P, "DRIFT_SCALE", 20.0))
                excess = max(0.0, drift_dist - deadband)
                w = max(0.0, min(1.0, excess / scale))
                anchor = (1.0 - w) * slow_anchor + w * F
            else:
                anchor = slow_anchor

            mr_eff = P.MR_STRENGTH * vol_factor
            dev = fv - anchor
            # Nonlinear boost: when |dev| exceeds BOOST_THRESHOLD, add extra
            # MR pressure proportional to excess. Steepens the target curve
            # at extremes so we hit POS_LIMIT faster on rare big deviations.
            #   boost = -BOOST_GAIN * sign(dev) * max(0, |dev| - threshold)
            # Linear in the excess; reduces to plain MR when threshold large.
            bt = getattr(P, "BOOST_THRESHOLD", 1e9)
            bg = getattr(P, "BOOST_GAIN", 0.0)
            boost = 0.0
            if bg > 0 and abs(dev) > bt:
                excess = abs(dev) - bt
                boost = -bg * (1.0 if dev > 0 else -1.0) * excess
            raw = (-mr_eff * dev
                   + boost
                   + getattr(P, 'OFI_GAIN', 0) * ofi_smooth
                   + range_comp)
            target = max(-P.POS_LIMIT, min(P.POS_LIMIT, int(round(raw))))

        decay = _profit_decay(P, pos, fv, avg_entry.get(P.SYMBOL, 0.0))
        target = int(round(target * decay))

        tox_factor = 1.0
        if P.TOX_FULL_OFF > 0:
            tox_factor = max(0.2, 1.0 - tox.get(P.SYMBOL, 0.0) / P.TOX_FULL_OFF)

        return target, self._do_orders(P, fv, pos, target, od, tox_factor)

    def _exec(self, state: TradingState, P, fv: float, target: int) -> List[Order]:
        od = state.order_depths[P.SYMBOL]
        pos = state.position.get(P.SYMBOL, 0)
        if not od.buy_orders or not od.sell_orders:
            return []
        return self._do_orders(P, fv, pos, target, od, 1.0)

    def _do_orders(self, P, fv: float, pos: int, target_pos: int,
                   od: OrderDepth, skew_factor: float = 1.0) -> List[Order]:
        orders: List[Order] = []
        skew = P.INV_SKEW * skew_factor * (pos - target_pos) / P.POS_LIMIT
        fv_eff = fv - skew

        best_bid = max(od.buy_orders.keys())
        best_ask = min(od.sell_orders.keys())
        buy_cap = P.POS_LIMIT - pos
        sell_cap = P.POS_LIMIT + pos

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
        base = getattr(P, "BASE_QTY", 10)
        bid_qty = min(buy_cap, max(0, target_pos - pos) + base)
        ask_qty = min(sell_cap, max(0, pos - target_pos) + base)

        if bid_qty > 0:
            orders.append(Order(P.SYMBOL, our_bid, bid_qty))
        if ask_qty > 0:
            orders.append(Order(P.SYMBOL, our_ask, -ask_qty))
        return orders
