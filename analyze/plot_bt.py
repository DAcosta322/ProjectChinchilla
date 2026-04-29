"""Generate logparse-style price/PnL graphs from BT log files in logs/.

Adapted from logparse.py: takes a directory containing BT .log files and
produces price + PnL graphs alongside them.

Usage:
    python plot_bt.py logs/<run_dir>
    python plot_bt.py logs/<run_dir>/r4_d3_*.log    # single file
"""
import json
import csv
import io
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Plot helpers (verbatim from logparse.py)
# ---------------------------------------------------------------------------
def plot_pnl(ax, product, pnl):
    if product in pnl:
        xs, ys = pnl[product]
        ax.plot(xs, ys, linewidth=1)
    ax.set_title(f"{product} PnL")
    ax.set_ylabel("P&L")
    ax.set_xlabel("Timestamp")
    ax.grid(True, alpha=0.3)


def plot_price(ax, product, mid, prices, annotate_qty=True):
    if product in mid:
        mxs, mys = mid[product]
        ax.plot(mxs, mys, color="gray", linewidth=0.8, alpha=0.6,
                label="mid", zorder=1)
    if product in prices:
        for label, color, marker in [("buy", "green", "^"),
                                     ("sell", "red", "v"),
                                     ("self", "blue", "o")]:
            txs, tys, tqs = prices[product][label]
            if not txs:
                continue
            max_q = max(tqs); min_q = min(tqs)
            if max_q == min_q:
                alphas = [0.7] * len(tqs)
            else:
                alphas = [min(1.0, 0.2 + 0.8 * (q - min_q) / (max_q - min_q))
                          for q in tqs]
            ax.scatter(txs, tys, c=color, marker=marker, label=label,
                       alpha=alphas, s=20)
            if annotate_qty:
                for x, y, q in zip(txs, tys, tqs):
                    ax.annotate(str(q), (x, y), textcoords="offset points",
                                xytext=(0, 6 if marker == "^" else -10),
                                fontsize=6, ha="center", color=color,
                                alpha=0.8)
        if any(prices[product][k][0] for k in ("buy", "sell", "self")):
            ax.legend(fontsize=8, loc="best")
    ax.set_title(f"{product} Price")
    ax.set_ylabel("Price"); ax.set_xlabel("Timestamp")
    ax.grid(True, alpha=0.3)


STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
DELTA1 = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]


def parse_log(log_path: Path):
    """Returns (pnl, mid, prices, summary). Same shape as logparse uses."""
    log = json.loads(log_path.read_text())
    pnl: dict = {}
    mid: dict = {}
    prices: dict = {}

    al = log.get("activitiesLog", "")
    if isinstance(al, list):
        # In case it's structured already
        rows = al
    else:
        rows = list(csv.DictReader(io.StringIO(al), delimiter=";"))
    for row in rows:
        product = row["product"]
        ts = int(row["timestamp"])
        try:
            y = float(row["profit_and_loss"])
        except (KeyError, ValueError):
            continue
        pnl.setdefault(product, ([], []))
        pnl[product][0].append(ts); pnl[product][1].append(y)
        mp = row.get("mid_price", "")
        if mp and float(mp) != 0:
            mid.setdefault(product, ([], []))
            mid[product][0].append(ts); mid[product][1].append(float(mp))

    for t in log.get("tradeHistory", []):
        sym = t["symbol"]
        prices.setdefault(sym, {"buy": ([], [], []), "sell": ([], [], []),
                                "self": ([], [], [])})
        ts = t["timestamp"]; px = t["price"]; qty = t["quantity"]
        if px == 0 or qty == 0:
            continue
        is_buyer = t.get("buyer") == "SUBMISSION"
        is_seller = t.get("seller") == "SUBMISSION"
        if is_buyer and is_seller:
            key = "self"
        elif is_buyer:
            key = "buy"
        elif is_seller:
            key = "sell"
        else:
            continue
        prices[sym][key][0].append(ts)
        prices[sym][key][1].append(px)
        prices[sym][key][2].append(qty)

    return pnl, mid, prices


def plot_one(log_path: Path, out_dir: Path, label: str):
    pnl, mid, prices = parse_log(log_path)
    saved = []

    # Delta-1 panels
    for product in DELTA1:
        if product not in pnl:
            continue
        fig, (ax_pnl, ax_price) = plt.subplots(1, 2, figsize=(14, 5))
        plot_pnl(ax_pnl, product, pnl)
        plot_price(ax_price, product, mid, prices, annotate_qty=False)
        fig.suptitle(f"{product} — {label}", fontsize=13)
        fig.tight_layout()
        out = out_dir / f"{label}_{product}.png"
        fig.savefig(out, dpi=130)
        saved.append(out)
        plt.close(fig)

    # Voucher grid
    vouchers = [v for v in STRIKES if v in pnl]
    vouchers.sort(key=lambda v: STRIKES[v])
    if vouchers:
        n = len(vouchers)
        ncols = 5 if n > 5 else n
        nrows = (n + ncols - 1) // ncols

        fig_p, axes_p = plt.subplots(nrows, ncols,
                                     figsize=(4 * ncols, 3.2 * nrows),
                                     squeeze=False)
        for i, v in enumerate(vouchers):
            ax = axes_p[i // ncols][i % ncols]
            plot_price(ax, v, mid, prices, annotate_qty=False)
            ax.set_title(f"{v} (K={STRIKES[v]}) Price", fontsize=10)
        for i in range(n, nrows * ncols):
            axes_p[i // ncols][i % ncols].axis("off")
        fig_p.suptitle(f"VEV Vouchers Price — {label}", fontsize=13)
        fig_p.tight_layout()
        out = out_dir / f"{label}_vouchers_price.png"
        fig_p.savefig(out, dpi=130)
        saved.append(out)
        plt.close(fig_p)

        fig_q, axes_q = plt.subplots(nrows, ncols,
                                     figsize=(4 * ncols, 3.2 * nrows),
                                     squeeze=False)
        for i, v in enumerate(vouchers):
            ax = axes_q[i // ncols][i % ncols]
            plot_pnl(ax, v, pnl)
            ax.set_title(f"{v} (K={STRIKES[v]}) PnL", fontsize=10)
        for i in range(n, nrows * ncols):
            axes_q[i // ncols][i % ncols].axis("off")
        fig_q.suptitle(f"VEV Vouchers PnL — {label}", fontsize=13)
        fig_q.tight_layout()
        out = out_dir / f"{label}_vouchers_pnl.png"
        fig_q.savefig(out, dpi=130)
        saved.append(out)
        plt.close(fig_q)

    return saved


def main():
    if len(sys.argv) < 2:
        print("usage: python plot_bt.py logs/<run_dir>")
        sys.exit(1)
    target = Path(sys.argv[1])
    if target.is_file():
        files = [target]
        out_dir = target.parent
    else:
        files = sorted(target.glob("*.log"))
        out_dir = target
    if not files:
        print(f"no .log files found in {target}")
        sys.exit(1)
    for f in files:
        # label from filename: r4_dN_<algo>_<id>.log -> r4_dN
        parts = f.stem.split("_")
        label = "_".join(parts[:2]) if len(parts) >= 2 else f.stem
        print(f"plotting {f.name} -> label={label}")
        saved = plot_one(f, out_dir, label)
        for s in saved:
            print(f"  saved {s}")


if __name__ == "__main__":
    main()
