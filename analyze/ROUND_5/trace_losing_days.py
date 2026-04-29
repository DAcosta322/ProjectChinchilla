"""Per-product per-day trajectory plots for v4e single-day losers.

For each (day, product) in TARGETS, plot 3 panels:
  - mid price + buy/sell fill marks
  - signed position over time
  - cumulative PnL

Helps see WHEN the loss accumulates and HOW the position correlates with mid.

Output: data/ROUND_5/loser_traces/{day}_{product}.png
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import backtester as BT  # noqa: E402

ALGO = "ROUND_5/OTHERS/v4e_skip_chronic"
ROUND = 5

# Non-PEBBLES, non-{LAMB_WOOL/DISHES/DARK_MATTER} single-day losses >= $1K.
TARGETS = [
    (2, "OXYGEN_SHAKE_MINT",          -3994),
    (3, "MICROCHIP_TRIANGLE",         -2932),
    (3, "TRANSLATOR_ECLIPSE_CHARCOAL", -1554),
    (2, "GALAXY_SOUNDS_SOLAR_FLAMES", -1345),
    (2, "GALAXY_SOUNDS_BLACK_HOLES",  -1338),
    (4, "ROBOT_IRONING",              -1194),
    (4, "MICROCHIP_SQUARE",           -1000),
]

OUT_DIR = REPO_ROOT / "data" / "ROUND_5" / "loser_traces"


def run_day(module, day):
    reader = BT.DataReader(REPO_ROOT / "data")
    with contextlib.redirect_stdout(io.StringIO()):
        return BT.run_backtest(module, reader, ROUND, day)


def position_trajectory(trade_history, product):
    """Build (timestamps, signed_position) from own trades on a product."""
    trades = []
    for t in trade_history:
        if t.get("symbol") != product:
            continue
        is_buyer = t.get("buyer") == "SUBMISSION"
        is_seller = t.get("seller") == "SUBMISSION"
        if is_buyer and is_seller:
            continue
        if not (is_buyer or is_seller):
            continue
        signed_qty = t["quantity"] if is_buyer else -t["quantity"]
        trades.append((t["timestamp"], signed_qty, t["price"], t["quantity"], is_buyer))
    trades.sort()
    ts, pos = [0], [0]
    cur = 0
    for tp, sq, px, q, ib in trades:
        cur += sq
        ts.append(tp)
        pos.append(cur)
    return ts, pos, trades


def plot_target(day, product, expected_pnl, bt_result, mids, run_dir):
    th = bt_result.get("log", {}).get("tradeHistory", [])
    pos_ts, pos_vals, trades = position_trajectory(th, product)
    pnl_ts_map = bt_result.get("pnl_by_prod_at_ts", {}).get(product, {})
    pnl_ts = sorted(pnl_ts_map.keys())
    pnl_vals = [pnl_ts_map[t] for t in pnl_ts]

    mid_ts = [t for t, _ in mids]
    mid_vals = [m for _, m in mids]

    fig, axes = plt.subplots(3, 1, figsize=(15, 9), sharex=True)
    ax_mid, ax_pos, ax_pnl = axes

    # Panel 1: mid + fills
    ax_mid.plot(mid_ts, mid_vals, color="gray", lw=0.7, alpha=0.7, label="mid")
    buys  = [(tp, px, q) for tp, sq, px, q, ib in trades if ib]
    sells = [(tp, px, q) for tp, sq, px, q, ib in trades if not ib]
    if buys:
        ax_mid.scatter([b[0] for b in buys],  [b[1] for b in buys],
                       c="green", marker="^", s=18, alpha=0.6, label="our buys")
    if sells:
        ax_mid.scatter([s[0] for s in sells], [s[1] for s in sells],
                       c="red", marker="v", s=18, alpha=0.6, label="our sells")
    ax_mid.set_ylabel("mid / fill px")
    ax_mid.set_title(
        f"{product}  D{day}  (PnL ${expected_pnl})  "
        f"drift={mid_vals[-1]-mid_vals[0]:+.0f}  range={max(mid_vals)-min(mid_vals):.0f}",
        fontsize=11)
    ax_mid.grid(True, alpha=0.3)
    ax_mid.legend(fontsize=8, loc="best")

    # Panel 2: position
    ax_pos.fill_between(pos_ts, 0, pos_vals, where=[v > 0 for v in pos_vals],
                        color="green", alpha=0.3, step="post")
    ax_pos.fill_between(pos_ts, 0, pos_vals, where=[v < 0 for v in pos_vals],
                        color="red", alpha=0.3, step="post")
    ax_pos.step(pos_ts, pos_vals, where="post", color="black", lw=0.8)
    ax_pos.axhline(0, color="black", lw=0.5)
    ax_pos.axhline(+10, color="gray", lw=0.4, ls="--")
    ax_pos.axhline(-10, color="gray", lw=0.4, ls="--")
    ax_pos.set_ylabel("position")
    ax_pos.grid(True, alpha=0.3)

    # Panel 3: cumulative PnL
    ax_pnl.plot(pnl_ts, pnl_vals, color="purple", lw=1.2)
    ax_pnl.fill_between(pnl_ts, 0, pnl_vals, where=[v < 0 for v in pnl_vals],
                        color="red", alpha=0.2)
    ax_pnl.fill_between(pnl_ts, 0, pnl_vals, where=[v > 0 for v in pnl_vals],
                        color="green", alpha=0.2)
    ax_pnl.axhline(0, color="black", lw=0.5)
    ax_pnl.set_ylabel("cum PnL ($)")
    ax_pnl.set_xlabel("timestamp")
    ax_pnl.grid(True, alpha=0.3)

    fig.tight_layout()
    out = run_dir / f"d{day}_{product}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, pos_vals, pnl_vals


def load_day_mids(day, product):
    import csv
    path = REPO_ROOT / "data" / "ROUND_5" / f"prices_round_5_day_{day}.csv"
    out = []
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter=";")
        for row in rd:
            if row["product"] != product:
                continue
            try:
                out.append((int(row["timestamp"]), float(row["mid_price"])))
            except (ValueError, TypeError):
                continue
    out.sort()
    return out


def fragment_pnl(pnl_ts_map, frag_ticks=100000):
    if not pnl_ts_map:
        return []
    tss = sorted(pnl_ts_map.keys())
    out = []
    cur = tss[0]
    end = tss[-1]
    while cur <= end:
        nxt = cur + frag_ticks
        if cur == tss[0]:
            start_pnl = 0.0
        else:
            prior = max((t for t in tss if t < cur), default=None)
            start_pnl = pnl_ts_map[prior] if prior else 0.0
        last = max((t for t in tss if t < nxt), default=None)
        if last is None:
            break
        out.append((cur, last, pnl_ts_map[last] - start_pnl))
        cur = nxt
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    algo_path = REPO_ROOT / "algorithms" / f"{ALGO}.py"
    spec = importlib.util.spec_from_file_location("trader_algo", algo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    days_needed = sorted({d for d, _, _ in TARGETS})
    results = {}
    for d in days_needed:
        print(f"running BT day {d} ...")
        results[d] = run_day(module, d)

    summary = []
    for day, product, expected in TARGETS:
        print(f"\n--- {product} D{day} ---")
        mids = load_day_mids(day, product)
        out, pos_vals, pnl_vals = plot_target(
            day, product, expected, results[day], mids, OUT_DIR
        )
        # Stats
        max_pos = max(pos_vals) if pos_vals else 0
        min_pos = min(pos_vals) if pos_vals else 0
        max_pnl = max(pnl_vals) if pnl_vals else 0
        min_pnl = min(pnl_vals) if pnl_vals else 0
        # PnL drawdown timing
        peak_t = pnl_vals.index(max_pnl) if pnl_vals else 0
        trough_t = pnl_vals.index(min_pnl) if pnl_vals else 0
        peak_ts = sorted(results[day]["pnl_by_prod_at_ts"][product].keys())[peak_t]
        trough_ts = sorted(results[day]["pnl_by_prod_at_ts"][product].keys())[trough_t]
        # Fragment PnL
        frag_pnls = fragment_pnl(results[day]["pnl_by_prod_at_ts"][product])
        worst_frag = min(frag_pnls, key=lambda x: x[2]) if frag_pnls else None
        summary.append({
            "day": day, "product": product, "loss": expected,
            "pos_max": max_pos, "pos_min": min_pos,
            "pnl_peak": max_pnl, "pnl_peak_ts": peak_ts,
            "pnl_trough": min_pnl, "pnl_trough_ts": trough_ts,
            "worst_frag": worst_frag,
        })
        print(f"  max_pos={max_pos:+}  min_pos={min_pos:+}  "
              f"peak_pnl={max_pnl:+.0f} @ ts={peak_ts}  "
              f"trough_pnl={min_pnl:+.0f} @ ts={trough_ts}")
        if worst_frag:
            print(f"  worst frag [{worst_frag[0]}..{worst_frag[1]}]: ${worst_frag[2]:+.0f}")
        print(f"  saved {out}")

    # Print summary table
    print("\n\n=== SUMMARY ===")
    print(f"{'day':>3} {'product':30s} {'loss':>8} {'pos_range':>11} "
          f"{'peak/trough':>16} {'worst_frag':>30}")
    for s in summary:
        rng = f"{s['pos_min']:+}..{s['pos_max']:+}"
        pt = f"{s['pnl_peak']:+.0f}/{s['pnl_trough']:+.0f}"
        wf = (f"[{s['worst_frag'][0]}..{s['worst_frag'][1]}]: "
              f"${s['worst_frag'][2]:+.0f}") if s['worst_frag'] else ""
        print(f"D{s['day']:>2} {s['product']:30s} {s['loss']:>+8} {rng:>11} {pt:>16} {wf:>30}")


if __name__ == "__main__":
    main()
