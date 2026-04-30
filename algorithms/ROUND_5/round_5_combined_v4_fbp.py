"""v4 — pebbles_v12 + v3_skip_cd OTHER_8 + SNACKPACK upgrades.

PEBBLES from pebbles_v12.py:
- Per-leg DEV_GAIN (XS=25, S=10, M=0, L=0, XL=25)
- Per-leg MIN_SPREAD gate (XS=8, S=2, M=14, L=12, XL=20) — skip MM in narrow spreads
- MM_QTY=7
- v12 standalone BT: $122,896

OTHER_8 from v3_skip_cd:
- 5-tick same-side cooldown after fills
- Sign-fixed (buyer/seller fields), ts-deduped own_trades processing
- SKIP_MM = {SLEEP_POD_LAMB_WOOL, ROBOT_DISHES}

SNACKPACK upgrades (vs v3_skip_cd):
- Runtime alpha computation (HL_MEAN/HL_VAR sweep-patchable)
- Fill-VWAP anchor when |pos|>=6 (BT +$683)
- Multi-rung exit ladder when |pos|>=8 (BT +$122)
- Asymmetric Z_FULL (entry vs exit) — disabled by default (no BT gain)
- Soft-clip target — disabled by default (LOSES BT)
- Drift detector — disabled by default (no BT gain)
- PISTA-tied-to-STRAW (+$12.9K vs imbalance MM)
- sp_imb_pista state removed
- PISTA_LEV at extreme z_B — disabled by default (target saturates ±10)
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
PEBBLES_OTHERS = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]
PEBBLES_BASKET_SUM = 50000

SNACK_CHOC  = "SNACKPACK_CHOCOLATE"
SNACK_VAN   = "SNACKPACK_VANILLA"
SNACK_PISTA = "SNACKPACK_PISTACHIO"
SNACK_STRAW = "SNACKPACK_STRAWBERRY"
SNACK_RASP  = "SNACKPACK_RASPBERRY"
SNACKPACK = [SNACK_CHOC, SNACK_VAN, SNACK_PISTA, SNACK_STRAW, SNACK_RASP]
SNACK_PAIRS = [SNACK_CHOC, SNACK_VAN, SNACK_STRAW, SNACK_RASP]

OTHER_8 = [
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
    "ROBOT_LAUNDRY", "ROBOT_IRONING",
    "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
    "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
]

SKIP_MM = {"SLEEP_POD_LAMB_WOOL", "ROBOT_DISHES", "GALAXY_SOUNDS_DARK_MATTER"}

# Per-product MIN_SPREAD for OTHER_8 (only quote when spread >= threshold).
# Default 0 = no gate. Reserved for future product-specific tuning;
# DISHES gate=8 was tested but lost -$2.4K BT (still skip entirely).
OTHER_8_MIN_SPREAD: dict = {}
OTHER_8_SET = set(OTHER_8)


POS_LIMIT = 10


class CD:
    O8_COOLDOWN_TICKS = 5
    O8_MM_QTY = 10


class P:
    """PEBBLES parameters from pebbles_v12.py."""
    MM_QTY = 7
    DEV_GAIN_XS = 25.0
    DEV_GAIN_S  = 10.0
    DEV_GAIN_M  = 0.0
    DEV_GAIN_L  = 0.0
    DEV_GAIN_XL = 25.0
    BASKET_ARB_QTY = 10
    MIN_SPREAD_XS = 8
    MIN_SPREAD_S  = 2
    MIN_SPREAD_M  = 14
    MIN_SPREAD_L  = 12
    MIN_SPREAD_XL = 20
    # Drift-stuck unwind: when |pos|>=UNWIND_GATE AND phantom wants to reduce
    # position AND quote px is at least UNWIND_EDGE ticks profitable vs recent
    # fill VWAP, allow take-profit quote in tight-spread regimes.
    UNWIND_GATE     = 10      # XL only, exact-limit gate
    UNWIND_QTY      = 5
    UNWIND_EDGE     = 8       # min profit vs fill VWAP
    UNWIND_TTL      = 200000  # fill VWAP TTL ticks (long horizon for XL drift)


class SP:
    HL_MEAN = 4000
    HL_VAR  = 8000
    Z_FULL_ENTRY = 0.2
    Z_FULL_EXIT  = 0.2
    Z_AGG = 2.5
    MIN_STD_A = 150.0
    MIN_STD_B = 200.0
    WARMUP_TICKS = 200
    MIN_GAP = 1

    SOFT_CLIP_GAIN = 1.0

    EXIT_LADDER_OFFSETS = (0, 1)
    EXIT_LADDER_QTY     = 2
    LADDER_GATE         = 8

    # Per-pair anchor gates (A = CHOC/VAN, B = STRAW/RASP). Sweep winner.
    ANCHOR_GATE_A     = 4
    ANCHOR_GATE_B     = 3
    ANCHOR_MIN_EDGE_A = 10
    ANCHOR_MIN_EDGE_B = 8
    ANCHOR_TTL_TICKS  = 10000
    # Backward-compat: ANCHOR_GATE is min(A,B) for trim-cutoff decisions
    ANCHOR_GATE       = 3

    HL_DRIFT_FAST    = 400
    DRIFT_THRESH     = 99.0
    DRIFT_QTY_FACTOR = 1.0

    PISTA_LEV        = 1.0
    PISTA_LEV_THRESH = 99.0

    # Factor B variant: f_B = STRAW + W_PISTA*PISTA - (1+W_PISTA)*RASP
    # W_PISTA=0 = baseline. Larger weight on PISTA averages it into the B-factor.
    FB_PISTA_W = 0.0


def _book(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


def _clip(x, lim):
    return max(-lim, min(lim, x))


def _peb_gain_for(p):
    return {
        "PEBBLES_XS": P.DEV_GAIN_XS,
        "PEBBLES_S":  P.DEV_GAIN_S,
        "PEBBLES_M":  P.DEV_GAIN_M,
        "PEBBLES_L":  P.DEV_GAIN_L,
        "PEBBLES_XL": P.DEV_GAIN_XL,
    }[p]


def _peb_min_spread_for(p):
    return {
        "PEBBLES_XS": P.MIN_SPREAD_XS,
        "PEBBLES_S":  P.MIN_SPREAD_S,
        "PEBBLES_M":  P.MIN_SPREAD_M,
        "PEBBLES_L":  P.MIN_SPREAD_L,
        "PEBBLES_XL": P.MIN_SPREAD_XL,
    }[p]


def _soft_clip_target(raw, gain, limit):
    if gain >= 1.0:
        return _clip(int(round(raw)), limit)
    import math
    sgn = 1.0 if raw >= 0 else -1.0
    mag = abs(raw)
    soft = limit * math.tanh((mag * gain) / limit)
    return _clip(int(round(sgn * soft)), limit)


class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        sp_ema = data.get("sp_ema", {})
        sp_var = data.get("sp_var", {})
        sp_ema_fast = data.get("sp_ema_fast", {})
        sp_n = data.get("sp_n", 0)
        cd_buy_until  = data.get("cd_buy_until", {})
        cd_sell_until = data.get("cd_sell_until", {})
        last_processed_ts = data.get("last_processed_ts", -1)
        anchor_fills = data.get("anchor_fills", {})

        alpha_m    = 1.0 - 0.5 ** (1.0 / SP.HL_MEAN)
        alpha_v    = 1.0 - 0.5 ** (1.0 / SP.HL_VAR)
        alpha_fast = 1.0 - 0.5 ** (1.0 / SP.HL_DRIFT_FAST)

        ts_now = state.timestamp
        cool_units = CD.O8_COOLDOWN_TICKS * 100

        max_seen_ts = last_processed_ts
        if hasattr(state, "own_trades") and state.own_trades:
            for p, trades in state.own_trades.items():
                if not trades:
                    continue
                for t in trades:
                    t_ts = getattr(t, "timestamp", 0)
                    if t_ts <= last_processed_ts:
                        continue
                    if t_ts > max_seen_ts:
                        max_seen_ts = t_ts
                    if p in OTHER_8_SET:
                        if getattr(t, "buyer", None) == "SUBMISSION":
                            cd_buy_until[p] = max(cd_buy_until.get(p, 0), ts_now + cool_units)
                        elif getattr(t, "seller", None) == "SUBMISSION":
                            cd_sell_until[p] = max(cd_sell_until.get(p, 0), ts_now + cool_units)
                    if p in SNACKPACK and SP.ANCHOR_GATE <= POS_LIMIT:
                        sgn = 1 if getattr(t, "buyer", None) == "SUBMISSION" else -1
                        qty = getattr(t, "quantity", 0)
                        px  = getattr(t, "price", 0)
                        anchor_fills.setdefault(p, []).append([t_ts, sgn * qty, px])
                    if p in PEBBLES:
                        sgn = 1 if getattr(t, "buyer", None) == "SUBMISSION" else -1
                        qty = getattr(t, "quantity", 0)
                        px  = getattr(t, "price", 0)
                        anchor_fills.setdefault(p, []).append([t_ts, sgn * qty, px])
        last_processed_ts = max_seen_ts

        # Trim expired anchor fills (use the longer of SNACK/PEBBLES TTLs per product)
        if anchor_fills:
            sp_cutoff  = ts_now - SP.ANCHOR_TTL_TICKS
            peb_cutoff = ts_now - P.UNWIND_TTL
            for p in list(anchor_fills.keys()):
                cutoff = peb_cutoff if p in PEBBLES else sp_cutoff
                anchor_fills[p] = [f for f in anchor_fills[p] if f[0] >= cutoff]
                if not anchor_fills[p]:
                    del anchor_fills[p]

        def fill_vwap(p):
            recs = anchor_fills.get(p, [])
            if not recs:
                return None
            tot_q = sum(r[1] for r in recs)
            tot_pxq = sum(r[1] * r[2] for r in recs)
            if tot_q == 0:
                return None
            return tot_pxq / tot_q

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
            return POS_LIMIT - state.position.get(p, 0) - buy_used.get(p, 0)

        def sell_cap(p):
            return POS_LIMIT + state.position.get(p, 0) - sell_used.get(p, 0)

        # ----- PEBBLES (pebbles_v12) -----
        def peb_pull_to_zero_mm(p, books_map, phantom_offset=0.0):
            r = books_map.get(p)
            if r is None:
                return
            bb, ba, _, _ = r
            if ba <= bb:
                return
            spread = ba - bb
            real_pos = state.position.get(p, 0)
            phantom = real_pos + phantom_offset
            mm_bid = bb + 1 if bb + 1 < ba else None
            mm_ask = ba - 1 if ba - 1 > bb else None
            spread_gated = spread <= _peb_min_spread_for(p)

            # Drift-stuck unwind: when at limit AND phantom wants opposite
            # direction, allow take-profit quote even in narrow spread
            # (bypasses MIN_SPREAD gate on UNWIND side only). Gated on
            # profitable VWAP edge to avoid firing on normal MR flips.
            # Sweep winner: all-legs gate=10, edge=8, qty=5, ttl=200000.
            stuck_long  = real_pos >=  P.UNWIND_GATE and phantom_offset > 0
            stuck_short = real_pos <= -P.UNWIND_GATE and phantom_offset < 0

            if spread_gated and not (stuck_long or stuck_short):
                return

            if phantom <= 0 and mm_bid is not None and not spread_gated:
                cap = min(P.MM_QTY, buy_cap(p))
                if cap > 0:
                    add(p, mm_bid, cap)
            if phantom >= 0 and mm_ask is not None and not spread_gated:
                cap = min(P.MM_QTY, sell_cap(p))
                if cap > 0:
                    add(p, mm_ask, -cap)

            # Stuck-pos unwind quotes (override spread gate)
            if stuck_long and mm_ask is not None:
                vwap = fill_vwap(p)
                if vwap is not None and mm_ask >= vwap + P.UNWIND_EDGE:
                    cap = min(P.UNWIND_QTY, sell_cap(p))
                    if cap > 0:
                        add(p, mm_ask, -cap)
            if stuck_short and mm_bid is not None:
                vwap = fill_vwap(p)
                if vwap is not None and mm_bid <= vwap - P.UNWIND_EDGE:
                    cap = min(P.UNWIND_QTY, buy_cap(p))
                    if cap > 0:
                        add(p, mm_bid, cap)

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
                offset = _peb_gain_for(p) * dev_XL
                peb_pull_to_zero_mm(p, peb_books, phantom_offset=offset)
            sum_ba = sum(peb_books[p][1] for p in PEBBLES)
            if sum_ba < PEBBLES_BASKET_SUM:
                qty = P.BASKET_ARB_QTY
                for p in PEBBLES:
                    qty = min(qty, peb_books[p][3], max(0, buy_cap(p)))
                if qty > 0:
                    for p in PEBBLES:
                        add(p, peb_books[p][1], qty)

        # ----- OTHER 8 -----
        def pull_to_zero_mm_cd(p, books_map, mm_qty=10):
            r = books_map.get(p)
            if r is None:
                return
            bb, ba, _, _ = r
            if ba <= bb:
                return
            min_spread = OTHER_8_MIN_SPREAD.get(p, 0)
            if min_spread > 0 and (ba - bb) < min_spread:
                return
            pos = state.position.get(p, 0)
            mm_bid = bb + 1 if bb + 1 < ba else None
            mm_ask = ba - 1 if ba - 1 > bb else None
            buy_blocked = ts_now < cd_buy_until.get(p, 0)
            sell_blocked = ts_now < cd_sell_until.get(p, 0)
            if pos <= 0 and mm_bid is not None and not buy_blocked:
                cap = min(mm_qty, buy_cap(p))
                if cap > 0:
                    add(p, mm_bid, cap)
            if pos >= 0 and mm_ask is not None and not sell_blocked:
                cap = min(mm_qty, sell_cap(p))
                if cap > 0:
                    add(p, mm_ask, -cap)

        other_books = {}
        for p in OTHER_8:
            if p in SKIP_MM:
                continue
            r = _book(state.order_depths.get(p))
            if r is not None:
                other_books[p] = r
        for p in OTHER_8:
            if p in SKIP_MM:
                continue
            pull_to_zero_mm_cd(p, other_books, mm_qty=CD.O8_MM_QTY)

        # ----- SNACKPACK -----
        sp_books = {}
        sp_mids = {}
        for p in SNACKPACK:
            r = _book(state.order_depths.get(p))
            if r is not None:
                sp_books[p] = r
                sp_mids[p] = (r[0] + r[1]) / 2.0

        min_std = {"A": SP.MIN_STD_A, "B": SP.MIN_STD_B}

        if all(p in sp_mids for p in SNACK_PAIRS):
            f_A = sp_mids[SNACK_CHOC] - sp_mids[SNACK_VAN]
            # f_B variant: include PISTA in long-side. Need PISTA available too.
            if SP.FB_PISTA_W > 0.0 and SNACK_PISTA in sp_mids:
                w = SP.FB_PISTA_W
                f_B = (sp_mids[SNACK_STRAW]
                       + w * sp_mids[SNACK_PISTA]
                       - (1.0 + w) * sp_mids[SNACK_RASP])
            else:
                f_B = sp_mids[SNACK_STRAW] - sp_mids[SNACK_RASP]
            factors = {"A": f_A, "B": f_B}
            for k, v in factors.items():
                if k not in sp_ema:
                    sp_ema[k] = v
                    sp_var[k] = min_std[k] ** 2
                    sp_ema_fast[k] = v
                else:
                    sp_ema[k] = sp_ema[k] + alpha_m * (v - sp_ema[k])
                    sp_var[k] = ((1 - alpha_v) * sp_var[k]
                                 + alpha_v * (v - sp_ema[k]) ** 2)
                    sp_ema_fast[k] = sp_ema_fast[k] + alpha_fast * (v - sp_ema_fast[k])
            sp_n += 1

            if sp_n >= SP.WARMUP_TICKS:
                sd_A = max(min_std["A"], sp_var["A"] ** 0.5)
                sd_B = max(min_std["B"], sp_var["B"] ** 0.5)
                z_A = (factors["A"] - sp_ema["A"]) / sd_A
                z_B = (factors["B"] - sp_ema["B"]) / sd_B

                drift_A = abs(sp_ema_fast["A"] - sp_ema["A"]) / sd_A
                drift_B = abs(sp_ema_fast["B"] - sp_ema["B"]) / sd_B
                qty_factor_A = SP.DRIFT_QTY_FACTOR if drift_A >= SP.DRIFT_THRESH else 1.0
                qty_factor_B = SP.DRIFT_QTY_FACTOR if drift_B >= SP.DRIFT_THRESH else 1.0

                scale_entry = POS_LIMIT / SP.Z_FULL_ENTRY
                scale_exit  = POS_LIMIT / SP.Z_FULL_EXIT

                def directional_target(z, pair_pos_sign):
                    raw_unsigned = -z
                    if pair_pos_sign == 0 or (raw_unsigned >= 0) == (pair_pos_sign > 0):
                        scale = scale_entry
                    else:
                        scale = scale_exit
                    raw = scale * raw_unsigned
                    return _soft_clip_target(raw, SP.SOFT_CLIP_GAIN, POS_LIMIT)

                pos_choc  = state.position.get(SNACK_CHOC, 0)
                pos_van   = state.position.get(SNACK_VAN, 0)
                pos_straw = state.position.get(SNACK_STRAW, 0)
                pos_rasp  = state.position.get(SNACK_RASP, 0)

                tgt_choc  = directional_target(z_A,  1 if pos_choc  > 0 else (-1 if pos_choc  < 0 else 0))
                tgt_van   = directional_target(-z_A, 1 if pos_van   > 0 else (-1 if pos_van   < 0 else 0))
                tgt_straw = directional_target(z_B,  1 if pos_straw > 0 else (-1 if pos_straw < 0 else 0))
                tgt_rasp  = directional_target(-z_B, 1 if pos_rasp  > 0 else (-1 if pos_rasp  < 0 else 0))

                target = {
                    SNACK_CHOC:  tgt_choc,
                    SNACK_VAN:   tgt_van,
                    SNACK_STRAW: tgt_straw,
                    SNACK_RASP:  tgt_rasp,
                }
                product_z = {SNACK_CHOC: abs(z_A), SNACK_VAN: abs(z_A),
                             SNACK_STRAW: abs(z_B), SNACK_RASP: abs(z_B)}
                pair_qty_factor = {SNACK_CHOC: qty_factor_A, SNACK_VAN: qty_factor_A,
                                   SNACK_STRAW: qty_factor_B, SNACK_RASP: qty_factor_B}

                def _anchor_params(p):
                    if p in (SNACK_CHOC, SNACK_VAN):
                        return SP.ANCHOR_GATE_A, SP.ANCHOR_MIN_EDGE_A
                    return SP.ANCHOR_GATE_B, SP.ANCHOR_MIN_EDGE_B

                def anchor_blocks_buy(p, px):
                    gate, edge = _anchor_params(p)
                    if abs(state.position.get(p, 0)) < gate:
                        return False
                    if state.position.get(p, 0) <= 0:
                        return False
                    vwap = fill_vwap(p)
                    if vwap is None:
                        return False
                    return px > vwap - edge

                def anchor_blocks_sell(p, px):
                    gate, edge = _anchor_params(p)
                    if abs(state.position.get(p, 0)) < gate:
                        return False
                    if state.position.get(p, 0) >= 0:
                        return False
                    vwap = fill_vwap(p)
                    if vwap is None:
                        return False
                    return px < vwap + edge

                def place_pair_orders(p, tgt, qf):
                    if p not in sp_books:
                        return
                    bb, ba, bv, av = sp_books[p]
                    pos = state.position.get(p, 0)
                    gap = tgt - pos
                    if abs(gap) < SP.MIN_GAP:
                        if abs(pos) >= SP.LADDER_GATE and SP.EXIT_LADDER_QTY > 0:
                            self._exit_ladder(p, pos, bb, ba, add, sell_cap, buy_cap)
                        return
                    aggressive = product_z[p] >= SP.Z_AGG
                    if gap > 0:
                        if aggressive:
                            fill = min(gap, av, buy_cap(p))
                            if fill > 0 and not anchor_blocks_buy(p, ba):
                                add(p, ba, fill)
                        rem = tgt - (pos + buy_used.get(p, 0))
                        if rem > 0:
                            px = bb + 1 if bb + 1 < ba else bb
                            cap = min(rem, buy_cap(p))
                            if qf < 1.0:
                                cap = max(0, int(cap * qf))
                            if cap > 0 and not anchor_blocks_buy(p, px):
                                add(p, px, cap)
                    else:
                        need = -gap
                        if aggressive:
                            fill = min(need, bv, sell_cap(p))
                            if fill > 0 and not anchor_blocks_sell(p, bb):
                                add(p, bb, -fill)
                        rem = (pos - sell_used.get(p, 0)) - tgt
                        if rem > 0:
                            px = ba - 1 if ba - 1 > bb else ba
                            cap = min(rem, sell_cap(p))
                            if qf < 1.0:
                                cap = max(0, int(cap * qf))
                            if cap > 0 and not anchor_blocks_sell(p, px):
                                add(p, px, -cap)
                    if abs(pos) >= SP.LADDER_GATE and SP.EXIT_LADDER_QTY > 0:
                        self._exit_ladder(p, pos, bb, ba, add, sell_cap, buy_cap)

                for p in SNACK_PAIRS:
                    place_pair_orders(p, target[p], pair_qty_factor[p])

                # PISTA-tied-to-STRAW (optional leverage at extreme z_B)
                if SNACK_PISTA in sp_books:
                    bb, ba, bv, av = sp_books[SNACK_PISTA]
                    pos = state.position.get(SNACK_PISTA, 0)
                    base_tgt = target[SNACK_STRAW]
                    if abs(z_B) >= SP.PISTA_LEV_THRESH and SP.PISTA_LEV != 1.0:
                        levered = int(round(SP.PISTA_LEV * base_tgt))
                        tgt_pista = _clip(levered, POS_LIMIT)
                    else:
                        tgt_pista = base_tgt
                    gap = tgt_pista - pos
                    if abs(gap) >= SP.MIN_GAP:
                        aggressive = product_z[SNACK_STRAW] >= SP.Z_AGG
                        if gap > 0:
                            if aggressive:
                                fill = min(gap, av, buy_cap(SNACK_PISTA))
                                if fill > 0:
                                    add(SNACK_PISTA, ba, fill)
                            rem = tgt_pista - (pos + buy_used.get(SNACK_PISTA, 0))
                            if rem > 0:
                                px = bb + 1 if bb + 1 < ba else bb
                                cap = min(rem, buy_cap(SNACK_PISTA))
                                if cap > 0:
                                    add(SNACK_PISTA, px, cap)
                        else:
                            need = -gap
                            if aggressive:
                                fill = min(need, bv, sell_cap(SNACK_PISTA))
                                if fill > 0:
                                    add(SNACK_PISTA, bb, -fill)
                            rem = (pos - sell_used.get(SNACK_PISTA, 0)) - tgt_pista
                            if rem > 0:
                                px = ba - 1 if ba - 1 > bb else ba
                                cap = min(rem, sell_cap(SNACK_PISTA))
                                if cap > 0:
                                    add(SNACK_PISTA, px, -cap)

        out_state = json.dumps({
            "sp_ema": sp_ema,
            "sp_var": sp_var,
            "sp_ema_fast": sp_ema_fast,
            "sp_n": sp_n,
            "cd_buy_until":  cd_buy_until,
            "cd_sell_until": cd_sell_until,
            "last_processed_ts": last_processed_ts,
            "anchor_fills": anchor_fills,
        })
        return orders, 0, out_state

    @staticmethod
    def _exit_ladder(p, pos, bb, ba, add, sell_cap, buy_cap):
        offsets = SP.EXIT_LADDER_OFFSETS
        qty = SP.EXIT_LADDER_QTY
        if qty <= 0:
            return
        if pos > 0:
            for off in offsets:
                if off == 0:
                    continue
                px = ba - 1 + off
                if px <= bb or px >= ba + 5:
                    continue
                cap = min(qty, sell_cap(p))
                if cap > 0:
                    add(p, px, -cap)
        elif pos < 0:
            for off in offsets:
                if off == 0:
                    continue
                px = bb + 1 - off
                if px >= ba or px <= bb - 5:
                    continue
                cap = min(qty, buy_cap(p))
                if cap > 0:
                    add(p, px, cap)
