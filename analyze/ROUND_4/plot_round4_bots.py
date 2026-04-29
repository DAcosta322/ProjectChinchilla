"""Plot each bot's trade activity in round 4.

Outputs to dump/round4_bots/:
  - per_bot_<Mark>.png   : 1 row per product the bot traded; mid line + buy/sell scatter
  - product_<X>.png      : mid line + every bot's trades color-coded
  - net_position.png     : cumulative net position per (bot, product) over time
"""
from __future__ import annotations
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/ROUND_4")
OUT = Path("dump/round4_bots")
OUT.mkdir(parents=True, exist_ok=True)
DAYS = [1, 2, 3]

# Stable color per bot
BOT_COLORS = {
    "Mark 01": "#1f77b4",   # blue   (voucher buyer)
    "Mark 14": "#2ca02c",   # green  (passive MM)
    "Mark 22": "#d62728",   # red    (voucher seller)
    "Mark 38": "#ff7f0e",   # orange (aggressor)
    "Mark 49": "#9467bd",   # purple
    "Mark 55": "#8c564b",   # brown  (VELVET aggressor)
    "Mark 67": "#e377c2",   # pink   (VELVET whale)
}


def load_all():
    mids = defaultdict(dict)   # product -> {ts_global: mid}
    trades = []                # list of dicts with ts_global, day, ts, ...
    for d in DAYS:
        offset = (d - 1) * 1_000_000
        with open(DATA / f"prices_round_4_day_{d}.csv") as f:
            for row in csv.DictReader(f, delimiter=";"):
                try:
                    m = float(row["mid_price"])
                except (TypeError, ValueError):
                    continue
                ts_g = int(row["timestamp"]) + offset
                mids[row["product"]][ts_g] = m
        with open(DATA / f"trades_round_4_day_{d}.csv") as f:
            for row in csv.DictReader(f, delimiter=";"):
                ts = int(row["timestamp"])
                trades.append({
                    "day": d,
                    "ts": ts,
                    "ts_global": ts + offset,
                    "buyer": row["buyer"],
                    "seller": row["seller"],
                    "symbol": row["symbol"],
                    "price": float(row["price"]),
                    "qty": int(float(row["quantity"])),
                })
    return mids, trades


def add_day_dividers(ax, ymin, ymax):
    for d in DAYS[:-1]:
        ax.axvline(x=d * 1_000_000, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)


def plot_per_bot(mids, trades):
    """For each bot, one figure with one row per product the bot traded.
    Mid line + buy(green up-triangle) / sell(red down-triangle) scatter, size~qty.
    """
    bot_prod_trades = defaultdict(lambda: defaultdict(list))
    for t in trades:
        sym = t["symbol"]
        if t["buyer"] in BOT_COLORS:
            bot_prod_trades[t["buyer"]][sym].append((t["ts_global"], t["price"], t["qty"], "buy"))
        if t["seller"] in BOT_COLORS:
            bot_prod_trades[t["seller"]][sym].append((t["ts_global"], t["price"], t["qty"], "sell"))

    for bot, prod_trades in bot_prod_trades.items():
        # Sort products by total qty
        prods = sorted(prod_trades.keys(),
                       key=lambda p: -sum(t[2] for t in prod_trades[p]))
        prods = prods[:8]  # top 8 by volume
        n = len(prods)
        fig, axes = plt.subplots(n, 1, figsize=(14, 2.4 * n), sharex=True)
        if n == 1:
            axes = [axes]
        fig.suptitle(f"{bot} — trade activity (size ∝ qty)",
                     fontsize=12, fontweight="bold")
        for ax, sym in zip(axes, prods):
            mid_pts = sorted(mids[sym].items())
            xs = [t for t, _ in mid_pts]
            ys = [m for _, m in mid_pts]
            ax.plot(xs, ys, color="black", linewidth=0.6, alpha=0.5, label="mid")
            buys = [(ts, p, q) for ts, p, q, side in prod_trades[sym] if side == "buy"]
            sells = [(ts, p, q) for ts, p, q, side in prod_trades[sym] if side == "sell"]
            if buys:
                ax.scatter([b[0] for b in buys], [b[1] for b in buys],
                           s=[max(8, b[2] * 4) for b in buys],
                           marker="^", color="green", alpha=0.7,
                           edgecolors="black", linewidths=0.3, label=f"buy ({len(buys)})")
            if sells:
                ax.scatter([s[0] for s in sells], [s[1] for s in sells],
                           s=[max(8, s[2] * 4) for s in sells],
                           marker="v", color="red", alpha=0.7,
                           edgecolors="black", linewidths=0.3, label=f"sell ({len(sells)})")
            ax.set_ylabel(sym, fontsize=8)
            ax.legend(loc="best", fontsize=7, framealpha=0.7)
            ax.grid(True, alpha=0.3)
            if ys:
                add_day_dividers(ax, min(ys), max(ys))
        axes[-1].set_xlabel("timestamp (continuous across days 1-3, dotted = day boundary)")
        plt.tight_layout()
        out = OUT / f"per_bot_{bot.replace(' ', '_')}.png"
        plt.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_per_product(mids, trades):
    """For each major product, one figure showing mid + every bot's trades
    color-coded by bot. Buys = up-triangle, sells = down-triangle."""
    prod_volume = defaultdict(int)
    for t in trades:
        prod_volume[t["symbol"]] += t["qty"]
    prods = sorted(prod_volume.keys(), key=lambda p: -prod_volume[p])

    for sym in prods:
        prod_trades = [t for t in trades if t["symbol"] == sym]
        if not prod_trades:
            continue
        fig, ax = plt.subplots(figsize=(14, 5))
        mid_pts = sorted(mids[sym].items())
        xs = [t for t, _ in mid_pts]
        ys = [m for _, m in mid_pts]
        ax.plot(xs, ys, color="black", linewidth=0.6, alpha=0.5, label="mid")

        # One legend entry per bot per side
        plotted = set()
        for t in prod_trades:
            for who, marker, lbl_side in [(t["buyer"], "^", "buy"),
                                           (t["seller"], "v", "sell")]:
                if who not in BOT_COLORS:
                    continue
                key = (who, lbl_side)
                lbl = f"{who} {lbl_side}" if key not in plotted else None
                plotted.add(key)
                ax.scatter([t["ts_global"]], [t["price"]],
                           s=max(8, t["qty"] * 3),
                           marker=marker,
                           color=BOT_COLORS[who], alpha=0.55,
                           edgecolors="black", linewidths=0.2, label=lbl)
        ax.set_title(f"{sym} — all bot activity (size ∝ qty)", fontweight="bold")
        ax.set_xlabel("timestamp")
        ax.set_ylabel("price")
        ax.grid(True, alpha=0.3)
        if ys:
            add_day_dividers(ax, min(ys), max(ys))
        # Compress legend: 14 entries (7 bots x 2 sides)
        ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.85)
        plt.tight_layout()
        out = OUT / f"product_{sym}.png"
        plt.savefig(out, dpi=110)
        plt.close(fig)
        print(f"  wrote {out}")


def plot_net_positions(trades):
    """Cumulative net position per (bot, product) — shows directional drift."""
    series = defaultdict(list)  # (bot, sym) -> [(ts_global, net)]
    pos = defaultdict(int)
    for t in sorted(trades, key=lambda x: x["ts_global"]):
        for who, sign in [(t["buyer"], +1), (t["seller"], -1)]:
            if who not in BOT_COLORS:
                continue
            pos[(who, t["symbol"])] += sign * t["qty"]
            series[(who, t["symbol"])].append((t["ts_global"], pos[(who, t["symbol"])]))

    # Group by product
    by_sym = defaultdict(list)
    for (bot, sym), pts in series.items():
        # Only plot if final |net| >= 20 (filter out flat MM noise)
        if abs(pts[-1][1]) < 20:
            continue
        by_sym[sym].append((bot, pts))
    syms = sorted(by_sym.keys())

    n = len(syms)
    if n == 0:
        return
    cols = 2
    rows = (n + 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.6 * rows), sharex=True)
    axes = axes.flatten() if n > 1 else [axes]
    for ax, sym in zip(axes, syms):
        for bot, pts in by_sym[sym]:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.step(xs, ys, where="post", color=BOT_COLORS[bot],
                    label=f"{bot} (end={ys[-1]:+d})", linewidth=1.0)
        ax.set_title(sym, fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.legend(fontsize=7, loc="best", framealpha=0.7)
        ax.grid(True, alpha=0.3)
    for ax in axes[len(syms):]:
        ax.axis("off")
    fig.suptitle("Cumulative net position per bot per product (filter |net|>=20)",
                 fontweight="bold")
    plt.tight_layout()
    out = OUT / "net_position.png"
    plt.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    print("Loading data...")
    mids, trades = load_all()
    print(f"  {len(trades)} trades, {len(mids)} products")
    print("Plotting per-bot dashboards...")
    plot_per_bot(mids, trades)
    print("Plotting per-product overlays...")
    plot_per_product(mids, trades)
    print("Plotting cumulative net positions...")
    plot_net_positions(trades)
    print(f"Done. Outputs in {OUT}/")


if __name__ == "__main__":
    main()
