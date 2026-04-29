"""Per-product loss persistence + behavior analysis for the current best algo (v4c).

Steps:
  1. Run v4c BT for each day, collect per-product PnL.
  2. Build a persistence table: for each product, count days where PnL<0,
     also fragment-level loss persistence (10 fragments × 3 days = 30).
  3. For chronic losers, dump trade-side behavior:
       - own_trade fill counts (buy vs sell)
       - average fill price vs day's avg mid (adverse selection proxy)
       - day-end position
       - mid drift over the day
       - quoted-vs-filled price diff (where we paid vs the level we posted)

Output: data/ROUND_5/v4c_losers/
  per_product_pnl.csv          — product × day matrix
  per_product_fragment.csv     — product × (day, frag) matrix
  chronic_losers.txt           — text report
  per_product_behavior.csv     — fill diagnostics for losers
  losers_drift_vs_pnl.png      — scatter
"""

import contextlib
import csv
import importlib.util
import io
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import backtester as BT  # noqa: E402

import os
ALGO = os.environ.get("LOSER_ALGO", "ROUND_5/OTHERS/v4c_cooldown")
ROUND = 5
DAYS = [2, 3, 4]
FRAG_TICKS = 100000

OUT_TAG = ALGO.split("/")[-1]
OUT_DIR = REPO_ROOT / "data" / "ROUND_5" / f"{OUT_TAG}_losers"

CATEGORIES = {
    "GALAXY_SOUNDS": ["GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
                      "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
                      "GALAXY_SOUNDS_SOLAR_FLAMES"],
    "SLEEP_POD":    ["SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
                     "SLEEP_POD_NYLON", "SLEEP_POD_COTTON"],
    "MICROCHIP":    ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
                     "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE"],
    "PEBBLES":      ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"],
    "ROBOT":        ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
                     "ROBOT_LAUNDRY", "ROBOT_IRONING"],
    "UV_VISOR":     ["UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
                     "UV_VISOR_RED", "UV_VISOR_MAGENTA"],
    "TRANSLATOR":   ["TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
                     "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
                     "TRANSLATOR_VOID_BLUE"],
    "PANEL":        ["PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4"],
    "OXYGEN_SHAKE": ["OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
                     "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC"],
    "SNACKPACK":    ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
                     "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"],
}
ALL_PRODUCTS = [p for ms in CATEGORIES.values() for p in ms]
CAT_OF = {p: c for c, ms in CATEGORIES.items() for p in ms}


def fragment_pnls(pnl_at_ts, frag_ticks):
    if not pnl_at_ts:
        return []
    tss = sorted(pnl_at_ts.keys())
    out = []
    cur = tss[0]
    end_ts = tss[-1]
    while cur <= end_ts:
        nxt = cur + frag_ticks
        if cur == tss[0]:
            start_pnl = 0.0
        else:
            prior = max((t for t in tss if t < cur), default=None)
            start_pnl = pnl_at_ts[prior] if prior is not None else 0.0
        last_in = max((t for t in tss if t < nxt), default=None)
        if last_in is None:
            break
        out.append(pnl_at_ts[last_in] - start_pnl)
        cur = nxt
    return out


def run_v4c():
    """Run v4c BT for each day, return dict[day] = backtest_result."""
    algo_path = REPO_ROOT / "algorithms" / f"{ALGO}.py"
    spec = importlib.util.spec_from_file_location("trader_algo", algo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reader = BT.DataReader(REPO_ROOT / "data")
    out = {}
    for d in DAYS:
        with contextlib.redirect_stdout(io.StringIO()):
            r = BT.run_backtest(module, reader, ROUND, d)
        out[d] = r
    return out


def collect_market_data():
    """Per-day, per-product mid_price series from raw CSV (independent of BT)."""
    out = {d: {} for d in DAYS}
    for d in DAYS:
        path = REPO_ROOT / "data" / "ROUND_5" / f"prices_round_5_day_{d}.csv"
        with open(path, newline="") as f:
            rd = csv.DictReader(f, delimiter=";")
            for row in rd:
                p = row["product"]
                try:
                    ts = int(row["timestamp"])
                    m = float(row["mid_price"])
                except (ValueError, TypeError):
                    continue
                out[d].setdefault(p, []).append((ts, m))
        for p in out[d]:
            out[d][p].sort()
    return out


def own_trades_diagnostics(bt_result, mids):
    """Per (day, product), aggregate own-trade fill behavior.

    Returns dict[(day, product)] = {
       'n_buy', 'n_sell', 'qty_buy', 'qty_sell',
       'avg_buy_px', 'avg_sell_px', 'final_pos',
       'avg_mid', 'drift', 'avg_buy_vs_mid', 'avg_sell_vs_mid',
    }
    """
    out = {}
    own = bt_result.get("own_trades_by_product", {})
    if not own:
        # fallback: try trade history
        own = {}
        for t in bt_result.get("trade_history", []):
            sym = t.get("symbol")
            if sym is None:
                continue
            is_buyer = t.get("buyer") == "SUBMISSION"
            is_seller = t.get("seller") == "SUBMISSION"
            if is_buyer and is_seller:
                continue
            own.setdefault(sym, []).append({
                "buy": is_buyer,
                "price": t["price"],
                "quantity": t["quantity"],
                "timestamp": t["timestamp"],
            })
    return own


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Run v4c BT
    # ------------------------------------------------------------------
    print("running v4c BT for days 2/3/4 ...")
    results = run_v4c()

    # ------------------------------------------------------------------
    # 2. Per-product per-day PnL matrix
    # ------------------------------------------------------------------
    rows_pp = []
    for p in ALL_PRODUCTS:
        per_day = {d: float(results[d]["pnl_by_product"].get(p, 0.0)) for d in DAYS}
        total = sum(per_day.values())
        neg_days = sum(1 for v in per_day.values() if v < 0)
        rows_pp.append((CAT_OF[p], p, per_day, total, neg_days))

    with open(OUT_DIR / "per_product_pnl.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "product",
                    "d2_pnl", "d3_pnl", "d4_pnl", "total", "neg_days"])
        for cat, p, pd_, tot, neg in rows_pp:
            w.writerow([cat, p,
                        f"{pd_[2]:+.0f}", f"{pd_[3]:+.0f}", f"{pd_[4]:+.0f}",
                        f"{tot:+.0f}", neg])

    # ------------------------------------------------------------------
    # 3. Per-product per-fragment PnL (3 days × 10 frags = 30 fragments)
    # ------------------------------------------------------------------
    frag_table = {}  # product -> list[30] PnL per fragment
    for p in ALL_PRODUCTS:
        seq = []
        for d in DAYS:
            ts_pnl = results[d]["pnl_by_prod_at_ts"].get(p, {})
            seq.extend(fragment_pnls(ts_pnl, FRAG_TICKS))
        frag_table[p] = seq

    with open(OUT_DIR / "per_product_fragment.csv", "w", newline="") as f:
        w = csv.writer(f)
        header = ["category", "product"]
        for d in DAYS:
            for k in range(10):
                header.append(f"d{d}_f{k}")
        header += ["total", "n_neg_frags", "worst_frag", "frag_sum_negs"]
        w.writerow(header)
        for p in ALL_PRODUCTS:
            seq = frag_table[p]
            row = [CAT_OF[p], p] + [f"{v:+.0f}" for v in seq]
            tot = sum(seq)
            neg_n = sum(1 for v in seq if v < 0)
            worst = min(seq) if seq else 0
            neg_sum = sum(v for v in seq if v < 0)
            row += [f"{tot:+.0f}", neg_n, f"{worst:+.0f}", f"{neg_sum:+.0f}"]
            w.writerow(row)

    # ------------------------------------------------------------------
    # 4. Identify chronic losers + multi-day losers
    # ------------------------------------------------------------------
    chronic = [r for r in rows_pp if r[4] == 3]                # neg all 3 days
    multi   = [r for r in rows_pp if r[4] == 2]                # neg 2 of 3
    intermittent = [r for r in rows_pp if r[4] == 1]
    winners_only = [r for r in rows_pp if r[4] == 0]

    rows_pp_sorted = sorted(rows_pp, key=lambda r: r[3])

    # Drift per (product, day)
    print("loading raw mid data for drift calc ...")
    mids = collect_market_data()
    drifts = {}  # (product, day) -> end_mid - start_mid
    for d in DAYS:
        for p in ALL_PRODUCTS:
            if p in mids[d] and len(mids[d][p]) > 1:
                drifts[(p, d)] = mids[d][p][-1][1] - mids[d][p][0][1]

    # ------------------------------------------------------------------
    # 5. Trade-side behavior for chronic losers
    # ------------------------------------------------------------------
    behavior_rows = []
    for cat, p, per_day, tot, neg in rows_pp_sorted:
        for d in DAYS:
            day_pnl = per_day[d]
            if day_pnl >= 0:
                continue  # only analyze losing day-products
            # Pull own_trades from BT log (run_backtest stores them under log.tradeHistory)
            trades_for_p = []
            for t in results[d].get("log", {}).get("tradeHistory", []):
                if t.get("symbol") != p:
                    continue
                is_buyer = t.get("buyer") == "SUBMISSION"
                is_seller = t.get("seller") == "SUBMISSION"
                if is_buyer and is_seller:
                    continue
                if not (is_buyer or is_seller):
                    continue
                trades_for_p.append((
                    t["timestamp"],
                    "buy" if is_buyer else "sell",
                    t["price"], abs(t["quantity"]),
                ))
            if not trades_for_p:
                continue

            # avg mid for the day
            day_mids = mids[d].get(p, [])
            avg_mid = (
                sum(m for _, m in day_mids) / len(day_mids) if day_mids else float("nan")
            )

            n_buy = sum(1 for _, s, _, _ in trades_for_p if s == "buy")
            n_sell = sum(1 for _, s, _, _ in trades_for_p if s == "sell")
            qty_buy = sum(q for _, s, _, q in trades_for_p if s == "buy")
            qty_sell = sum(q for _, s, _, q in trades_for_p if s == "sell")
            avg_buy = (sum(px * q for _, s, px, q in trades_for_p if s == "buy")
                       / max(qty_buy, 1)) if qty_buy else float("nan")
            avg_sell = (sum(px * q for _, s, px, q in trades_for_p if s == "sell")
                        / max(qty_sell, 1)) if qty_sell else float("nan")
            net_qty = qty_buy - qty_sell  # >0 = net buyer = ended long
            drift = drifts.get((p, d), float("nan"))
            buy_vs_mid = (avg_buy - avg_mid) if not np.isnan(avg_buy) else float("nan")
            sell_vs_mid = (avg_sell - avg_mid) if not np.isnan(avg_sell) else float("nan")

            # adverse selection score: did we buy above mid (BAD) and sell below mid (BAD)?
            # we're a market maker, so we WANT buy<mid and sell>mid
            buy_adv = buy_vs_mid > 0  # bad
            sell_adv = sell_vs_mid < 0  # bad

            behavior_rows.append({
                "day": d, "category": cat, "product": p,
                "day_pnl": day_pnl,
                "n_buy": n_buy, "n_sell": n_sell,
                "qty_buy": qty_buy, "qty_sell": qty_sell,
                "net_qty": net_qty,
                "avg_buy_px": avg_buy, "avg_sell_px": avg_sell,
                "avg_mid": avg_mid,
                "buy_vs_mid": buy_vs_mid, "sell_vs_mid": sell_vs_mid,
                "drift": drift,
                "buy_adv": buy_adv, "sell_adv": sell_adv,
                "final_pos": results[d].get("position", {}).get(p, 0),
            })

    with open(OUT_DIR / "per_product_behavior.csv", "w", newline="") as f:
        if behavior_rows:
            cols = list(behavior_rows[0].keys())
            w = csv.writer(f)
            w.writerow(cols)
            for r in behavior_rows:
                row = []
                for c in cols:
                    v = r[c]
                    if isinstance(v, float):
                        row.append(f"{v:+.2f}" if not np.isnan(v) else "")
                    elif isinstance(v, bool):
                        row.append("1" if v else "0")
                    else:
                        row.append(v)
                w.writerow(row)

    # ------------------------------------------------------------------
    # 6. Text summary
    # ------------------------------------------------------------------
    lines = []
    total_bt = sum(results[d]["profit"] for d in DAYS)
    lines.append("=== v4c per-product loss persistence ===\n")
    lines.append(f"Total BT (3 days): ${total_bt:,.0f}\n\n")

    lines.append("--- Persistence buckets ---\n")
    lines.append(f"  Negative all 3 days (CHRONIC):   {len(chronic):>2} products\n")
    lines.append(f"  Negative 2 of 3 days (MULTI):    {len(multi):>2} products\n")
    lines.append(f"  Negative 1 of 3 days (INTERMITTENT): {len(intermittent):>2} products\n")
    lines.append(f"  Positive all 3 days (WINNERS):   {len(winners_only):>2} products\n\n")

    lines.append("--- Bottom 15 by total PnL ---\n")
    lines.append(f"  {'product':30s}  cat        D2       D3       D4    total  neg_days\n")
    for cat, p, per_day, tot, neg in rows_pp_sorted[:15]:
        lines.append(
            f"  {p:30s}  {cat[:10]:10s} {per_day[2]:+8.0f} {per_day[3]:+8.0f} "
            f"{per_day[4]:+8.0f} {tot:+8.0f}  {neg}\n"
        )

    lines.append("\n--- Top 15 by total PnL ---\n")
    lines.append(f"  {'product':30s}  cat        D2       D3       D4    total  neg_days\n")
    for cat, p, per_day, tot, neg in sorted(rows_pp, key=lambda r: -r[3])[:15]:
        lines.append(
            f"  {p:30s}  {cat[:10]:10s} {per_day[2]:+8.0f} {per_day[3]:+8.0f} "
            f"{per_day[4]:+8.0f} {tot:+8.0f}  {neg}\n"
        )

    # Chronic deeper-dive
    lines.append("\n--- CHRONIC LOSERS (negative all 3 days) ---\n")
    if not chronic:
        lines.append("  (none — losses are not persistent)\n")
    for cat, p, per_day, tot, neg in sorted(chronic, key=lambda r: r[3]):
        lines.append(f"  {p:30s}  {cat:14s}  total={tot:+.0f}\n")

    lines.append("\n--- MULTI-DAY LOSERS (2 of 3) ---\n")
    for cat, p, per_day, tot, neg in sorted(multi, key=lambda r: r[3]):
        lines.append(f"  {p:30s}  {cat:14s}  total={tot:+.0f}  D2/D3/D4="
                     f"{per_day[2]:+.0f}/{per_day[3]:+.0f}/{per_day[4]:+.0f}\n")

    # Per-product fragment loss persistence
    lines.append("\n--- Top 10 by negative-fragment count (out of 30) ---\n")
    frag_neg = [(p, sum(1 for v in frag_table[p] if v < 0),
                 min(frag_table[p]) if frag_table[p] else 0,
                 sum(v for v in frag_table[p] if v < 0))
                for p in ALL_PRODUCTS]
    frag_neg.sort(key=lambda x: -x[1])
    for p, n, worst, sumneg in frag_neg[:10]:
        lines.append(f"  {p:30s}  {n:>2}/30 frags neg  worst={worst:+.0f}  sumneg={sumneg:+.0f}\n")

    # Behavior for chronic + multi-day losers
    lines.append("\n--- Trade-side behavior (negative day-products only) ---\n")
    lines.append(f"  {'product':30s}  d  buys/sells  qty(b/s)  avg_buy_vs_mid  avg_sell_vs_mid  drift  net  pnl\n")
    for r in sorted(behavior_rows, key=lambda r: r["day_pnl"])[:20]:
        b_adv = "X" if r["buy_adv"] else "."
        s_adv = "X" if r["sell_adv"] else "."
        lines.append(
            f"  {r['product']:30s}  {r['day']}  "
            f"{r['n_buy']:>3}/{r['n_sell']:<3}  "
            f"{r['qty_buy']:>3}/{r['qty_sell']:<3}  "
            f"{r['buy_vs_mid']:+8.2f}{b_adv}  "
            f"{r['sell_vs_mid']:+8.2f}{s_adv}  "
            f"{r['drift']:+7.0f}  "
            f"{r['net_qty']:+4d}  "
            f"{r['day_pnl']:+8.0f}\n"
        )

    with open(OUT_DIR / "summary.txt", "w") as f:
        f.writelines(lines)

    # ------------------------------------------------------------------
    # 7. Plot: drift vs PnL across day-products (highlight losers)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, d in zip(axes, DAYS):
        xs, ys, names = [], [], []
        for p in ALL_PRODUCTS:
            x = drifts.get((p, d))
            y = float(results[d]["pnl_by_product"].get(p, 0.0))
            if x is None:
                continue
            xs.append(x)
            ys.append(y)
            names.append(p)
        colors = ["red" if y < 0 else "steelblue" for y in ys]
        ax.scatter(xs, ys, c=colors, s=60, alpha=0.7, edgecolors="black", linewidths=0.4)
        for x, y, n in zip(xs, ys, names):
            if y < -500 or abs(x) > 1500:
                ax.annotate(n.replace(CAT_OF[n] + "_", "")[:10],
                            (x, y), fontsize=7, alpha=0.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.axvline(0, color="black", lw=0.5)
        ax.set_xlabel("end_mid - start_mid (drift)")
        ax.set_ylabel("v4c product PnL")
        ax.set_title(f"Day {d}")
        ax.grid(True, alpha=0.3)
    fig.suptitle("v4c per-product PnL vs day drift  (red = loss)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "drift_vs_pnl.png", dpi=140)
    plt.close(fig)

    print(f"\nDone. Outputs in {OUT_DIR}")
    print(f"\n{'='*70}")
    with open(OUT_DIR / "summary.txt") as f:
        print(f.read())


if __name__ == "__main__":
    main()
