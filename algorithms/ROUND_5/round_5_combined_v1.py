"""Round 5 combined v1.

Three sub-strategies in one file:
  1. PEBBLES (5 products) — baseline_a: pull-to-zero one-sided MM + ETF arb
     at basket sum=50000. Validated +$1,276 platform sub 551400.
  2. SNACKPACK (5 products) — v5e: pair MR (CHOC/VAN, STRAW/RASP) z-score
     based on EMA of mid-difference + PISTA asymmetric MM. Memory says
     snackpack_v3 was the previous platform-tested base, v5e adds PISTA.
  3. Other 8 categories (40 products) — simple pull-to-zero MM with
     MM_QTY=10. No basket arb (independent products per correlation memory).
     Categories: GALAXY_SOUNDS, SLEEP_POD, MICROCHIP, ROBOT, UV_VISOR,
     TRANSLATOR, PANEL, OXYGEN_SHAKE.

State stored in traderData JSON under keys "snack_*" so PEBBLES and the
other-8 stay stateless.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import json


# ---------------------------------------------------------------------------
# Product definitions
# ---------------------------------------------------------------------------
PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]

SNACK_CHOC  = "SNACKPACK_CHOCOLATE"
SNACK_VAN   = "SNACKPACK_VANILLA"
SNACK_PISTA = "SNACKPACK_PISTACHIO"
SNACK_STRAW = "SNACKPACK_STRAWBERRY"
SNACK_RASP  = "SNACKPACK_RASPBERRY"
SNACKPACK = [SNACK_CHOC, SNACK_VAN, SNACK_PISTA, SNACK_STRAW, SNACK_RASP]
SNACK_PAIRS = [SNACK_CHOC, SNACK_VAN, SNACK_STRAW, SNACK_RASP]

OTHER_8 = [
    # GALAXY_SOUNDS
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    # SLEEP_POD
    "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    # MICROCHIP
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    # ROBOT
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
    "ROBOT_LAUNDRY", "ROBOT_IRONING",
    # UV_VISOR
    "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    # TRANSLATOR
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
    # PANEL
    "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    # OXYGEN_SHAKE
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
]


POS_LIMIT = 10  # all R5 products
PEBBLES_BASKET_SUM = 50000


# ---------------------------------------------------------------------------
# Snackpack v5e config (kept verbatim)
# ---------------------------------------------------------------------------
class SP:
    HL_MEAN = 4000
    HL_VAR  = 8000
    Z_FULL  = 0.2
    Z_AGG   = 2.5
    MIN_STD = {"A": 150.0, "B": 200.0}
    WARMUP_TICKS = 200
    MIN_GAP = 1
    HL_IMB = 200
    IMB_FULL = 1.0
    PISTA_MAIN_QTY = 8
    PISTA_OPP_QTY  = 5


SP_ALPHA_M   = 1.0 - 0.5 ** (1.0 / SP.HL_MEAN)
SP_ALPHA_V   = 1.0 - 0.5 ** (1.0 / SP.HL_VAR)
SP_ALPHA_IMB = 1.0 - 0.5 ** (1.0 / SP.HL_IMB)


def _book(od: OrderDepth):
    if not od or not od.buy_orders or not od.sell_orders:
        return None
    bb = max(od.buy_orders.keys())
    ba = min(od.sell_orders.keys())
    return bb, ba, od.buy_orders[bb], -od.sell_orders[ba]


def _clip(x, lim):
    return max(-lim, min(lim, x))


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------
class Trader:
    def run(self, state: TradingState):
        try:
            data = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            data = {}

        sp_ema = data.get("sp_ema", {})
        sp_var = data.get("sp_var", {})
        sp_n = data.get("sp_n", 0)
        sp_imb_pista = data.get("sp_imb_pista", 0.0)

        orders: Dict[str, List[Order]] = {}
        # Track per-product net order intent so position limit isn't violated
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

        # ---- Pull-to-zero MM helper used by PEBBLES + OTHER_8 ----
        def pull_to_zero_mm(p, books_map, mm_qty=10):
            r = books_map.get(p)
            if r is None:
                return
            bb, ba, _, _ = r
            if ba <= bb:
                return
            pos = state.position.get(p, 0)
            mm_bid = bb + 1 if bb + 1 < ba else None
            mm_ask = ba - 1 if ba - 1 > bb else None
            if pos <= 0 and mm_bid is not None:
                cap = min(mm_qty, buy_cap(p))
                if cap > 0:
                    add(p, mm_bid, cap)
            if pos >= 0 and mm_ask is not None:
                cap = min(mm_qty, sell_cap(p))
                if cap > 0:
                    add(p, mm_ask, -cap)

        # ====================================================================
        # 1. PEBBLES — baseline_a with ETF arb
        # ====================================================================
        peb_books = {}
        for p in PEBBLES:
            r = _book(state.order_depths.get(p))
            if r is not None:
                peb_books[p] = r

        if len(peb_books) == 5:
            # Per-product pull-to-zero MM
            for p in PEBBLES:
                pull_to_zero_mm(p, peb_books, mm_qty=10)
            # ETF basket arb (lock when sum-bid > 50000 or sum-ask < 50000)
            sum_bb = sum(peb_books[p][0] for p in PEBBLES)
            sum_ba = sum(peb_books[p][1] for p in PEBBLES)
            if sum_bb > PEBBLES_BASKET_SUM:
                qty = 10
                for p in PEBBLES:
                    qty = min(qty, peb_books[p][2], max(0, sell_cap(p)))
                if qty > 0:
                    for p in PEBBLES:
                        add(p, peb_books[p][0], -qty)
            if sum_ba < PEBBLES_BASKET_SUM:
                qty = 10
                for p in PEBBLES:
                    qty = min(qty, peb_books[p][3], max(0, buy_cap(p)))
                if qty > 0:
                    for p in PEBBLES:
                        add(p, peb_books[p][1], qty)

        # ====================================================================
        # 2. OTHER 8 categories — pure pull-to-zero MM, no basket arb
        # ====================================================================
        other_books = {}
        for p in OTHER_8:
            r = _book(state.order_depths.get(p))
            if r is not None:
                other_books[p] = r
        for p in OTHER_8:
            pull_to_zero_mm(p, other_books, mm_qty=10)

        # ====================================================================
        # 3. SNACKPACK — v5e (pair MR + PISTA imbalance MM)
        # ====================================================================
        sp_books = {}
        sp_mids = {}
        for p in SNACKPACK:
            r = _book(state.order_depths.get(p))
            if r is not None:
                sp_books[p] = r
                sp_mids[p] = (r[0] + r[1]) / 2.0

        if all(p in sp_mids for p in SNACK_PAIRS):
            # Pair factor mids
            f_A = sp_mids[SNACK_CHOC] - sp_mids[SNACK_VAN]
            f_B = sp_mids[SNACK_STRAW] - sp_mids[SNACK_RASP]
            factors = {"A": f_A, "B": f_B}

            for k, v in factors.items():
                if k not in sp_ema:
                    sp_ema[k] = v
                    sp_var[k] = SP.MIN_STD[k] ** 2
                else:
                    sp_ema[k] = sp_ema[k] + SP_ALPHA_M * (v - sp_ema[k])
                    sp_var[k] = ((1 - SP_ALPHA_V) * sp_var[k]
                                 + SP_ALPHA_V * (v - sp_ema[k]) ** 2)

            # PISTA imbalance EMA
            if SNACK_PISTA in sp_books:
                bb_p, ba_p, bv_p, av_p = sp_books[SNACK_PISTA]
                raw_imb = (bv_p - av_p) / max(bv_p + av_p, 1)
                sp_imb_pista = sp_imb_pista + SP_ALPHA_IMB * (raw_imb - sp_imb_pista)

            sp_n += 1

            # Trade only after warmup
            if sp_n >= SP.WARMUP_TICKS:

                def z(k):
                    sd = max(SP.MIN_STD[k], sp_var[k] ** 0.5)
                    return (factors[k] - sp_ema[k]) / sd

                z_A = z("A")
                z_B = z("B")
                scale = POS_LIMIT / SP.Z_FULL
                target = {
                    SNACK_CHOC:  _clip(int(round(-scale * z_A)), POS_LIMIT),
                    SNACK_VAN:   _clip(int(round(+scale * z_A)), POS_LIMIT),
                    SNACK_STRAW: _clip(int(round(-scale * z_B)), POS_LIMIT),
                    SNACK_RASP:  _clip(int(round(+scale * z_B)), POS_LIMIT),
                }
                product_z = {
                    SNACK_CHOC: abs(z_A), SNACK_VAN: abs(z_A),
                    SNACK_STRAW: abs(z_B), SNACK_RASP: abs(z_B),
                }

                # ---- Pair MR ----
                for p in SNACK_PAIRS:
                    if p not in sp_books:
                        continue
                    bb, ba, bv, av = sp_books[p]
                    pos = state.position.get(p, 0)
                    tgt = target[p]
                    gap = tgt - pos
                    if abs(gap) < SP.MIN_GAP:
                        continue
                    aggressive = product_z[p] >= SP.Z_AGG
                    if gap > 0:
                        if aggressive:
                            fill = min(gap, av, buy_cap(p))
                            if fill > 0:
                                add(p, ba, fill)
                        rem = tgt - (pos + buy_used.get(p, 0))
                        if rem > 0:
                            px = bb + 1 if bb + 1 < ba else bb
                            cap = min(rem, buy_cap(p))
                            if cap > 0:
                                add(p, px, cap)
                    else:
                        need = -gap
                        if aggressive:
                            fill = min(need, bv, sell_cap(p))
                            if fill > 0:
                                add(p, bb, -fill)
                        rem = (pos - sell_used.get(p, 0)) - tgt
                        if rem > 0:
                            px = ba - 1 if ba - 1 > bb else ba
                            cap = min(rem, sell_cap(p))
                            if cap > 0:
                                add(p, px, -cap)

                # ---- PISTA: asymmetric MM with imbalance lean ----
                if SNACK_PISTA in sp_books:
                    bb, ba, bv, av = sp_books[SNACK_PISTA]
                    pos = state.position.get(SNACK_PISTA, 0)
                    pista_scale = POS_LIMIT / SP.IMB_FULL
                    target_pista = _clip(
                        int(round(pista_scale * sp_imb_pista)), POS_LIMIT)
                    gap = target_pista - pos
                    if gap >= 0:
                        bid_q = SP.PISTA_MAIN_QTY
                        ask_q = SP.PISTA_OPP_QTY
                    else:
                        bid_q = SP.PISTA_OPP_QTY
                        ask_q = SP.PISTA_MAIN_QTY
                    if bb + 1 < ba and bid_q > 0:
                        cap = min(bid_q, buy_cap(SNACK_PISTA))
                        if cap > 0:
                            add(SNACK_PISTA, bb + 1, cap)
                    if ba - 1 > bb and ask_q > 0:
                        cap = min(ask_q, sell_cap(SNACK_PISTA))
                        if cap > 0:
                            add(SNACK_PISTA, ba - 1, -cap)

        # ---- Persist state ----
        out_state = json.dumps({
            "sp_ema": sp_ema,
            "sp_var": sp_var,
            "sp_n": sp_n,
            "sp_imb_pista": sp_imb_pista,
        })
        return orders, 0, out_state
