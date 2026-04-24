import json
import csv
import io
from pathlib import Path
import matplotlib.pyplot as plt

subnum = input("Subnum:")

dd = Path(__file__).parent / "dump" / subnum
jj = subnum + ".json"
ll = subnum + ".log"
j = json.loads((dd / jj).read_text())
log = json.loads((dd / ll).read_text())

print("PnL", j["profit"])
print("positions", j["positions"])
print("activities rows", len(j["activitiesLog"].splitlines()))

# trade history stats
trade = log["tradeHistory"]
print("trades", len(trade))
print("symbols", {s: sum(1 for t in trade if t["symbol"] == s) for s in set(t["symbol"] for t in trade)})

# parse activitiesLog CSV into per-product PNL series
reader = csv.DictReader(io.StringIO(j["activitiesLog"]), delimiter=";")
pnl = {}  # product -> (timestamps[], pnl[])
mid = {}  # product -> (timestamps[], mid_price[])
for row in reader:
    product = row["product"]
    ts = int(row["timestamp"])
    y = float(row["profit_and_loss"])
    if product not in pnl:
        pnl[product] = ([], [])
    pnl[product][0].append(ts)
    pnl[product][1].append(y)
    mp = row.get("mid_price", "")
    if mp and float(mp) != 0:
        if product not in mid:
            mid[product] = ([], [])
        mid[product][0].append(ts)
        mid[product][1].append(float(mp))

# parse trade history into buy/sell/self price series per product
prices = {}  # product -> {"buy": (ts[], px[], qty[]), "sell": ..., "self": ...}
for t in trade:
    sym = t["symbol"]
    if sym not in prices:
        prices[sym] = {"buy": ([], [], []), "sell": ([], [], []), "self": ([], [], [])}
    ts = t["timestamp"]
    px = t["price"]
    qty = t["quantity"]
    if px == 0 or qty == 0:
        continue
    is_buyer = t["buyer"] == "SUBMISSION"
    is_seller = t["seller"] == "SUBMISSION"
    if is_buyer and is_seller:
        prices[sym]["self"][0].append(ts)
        prices[sym]["self"][1].append(px)
        prices[sym]["self"][2].append(qty)
    elif is_buyer:
        prices[sym]["buy"][0].append(ts)
        prices[sym]["buy"][1].append(px)
        prices[sym]["buy"][2].append(qty)
    elif is_seller:
        prices[sym]["sell"][0].append(ts)
        prices[sym]["sell"][1].append(px)
        prices[sym]["sell"][2].append(qty)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_pnl(ax, product):
    if product in pnl:
        xs, ys = pnl[product]
        ax.plot(xs, ys, linewidth=1)
    ax.set_title(f"{product} PnL")
    ax.set_ylabel("P&L")
    ax.set_xlabel("Timestamp")
    ax.grid(True, alpha=0.3)


def plot_price(ax, product, annotate_qty=True):
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
            max_q = max(tqs)
            min_q = min(tqs)
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
    ax.set_ylabel("Price")
    ax.set_xlabel("Timestamp")
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
# Figure 1 + 2: delta-1 assets, each in their own window
# ---------------------------------------------------------------------------
DELTA1 = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
saved_paths = []

for product in DELTA1:
    if product not in pnl:
        continue
    fig, (ax_pnl, ax_price) = plt.subplots(1, 2, figsize=(14, 5))
    plot_pnl(ax_pnl, product)
    plot_price(ax_price, product, annotate_qty=True)
    fig.suptitle(f"{product} — Submission {subnum}", fontsize=13)
    fig.tight_layout()
    out = dd / f"{subnum}_{product}.png"
    fig.savefig(out, dpi=150)
    saved_paths.append(out)


# ---------------------------------------------------------------------------
# Figure 3: VEV vouchers — price panels arranged by strike
# ---------------------------------------------------------------------------
STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300, "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}
vouchers = [v for v in STRIKES if v in pnl]
vouchers.sort(key=lambda v: STRIKES[v])

if vouchers:
    n = len(vouchers)
    # 2 rows x 5 cols for 10 vouchers (or fewer); fall back for smaller n.
    ncols = 5 if n > 5 else n
    nrows = (n + ncols - 1) // ncols

    # Price grid
    fig_p, axes_p = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.2 * nrows),
                                 squeeze=False)
    for i, v in enumerate(vouchers):
        ax = axes_p[i // ncols][i % ncols]
        plot_price(ax, v, annotate_qty=False)
        ax.set_title(f"{v} (K={STRIKES[v]}) Price", fontsize=10)
    for i in range(n, nrows * ncols):
        axes_p[i // ncols][i % ncols].axis("off")
    fig_p.suptitle(f"VEV Vouchers Price — Submission {subnum}", fontsize=13)
    fig_p.tight_layout()
    out = dd / f"{subnum}_vouchers_price.png"
    fig_p.savefig(out, dpi=150)
    saved_paths.append(out)

    # PnL grid
    fig_q, axes_q = plt.subplots(nrows, ncols,
                                 figsize=(4 * ncols, 3.2 * nrows),
                                 squeeze=False)
    for i, v in enumerate(vouchers):
        ax = axes_q[i // ncols][i % ncols]
        plot_pnl(ax, v)
        ax.set_title(f"{v} (K={STRIKES[v]}) PnL", fontsize=10)
    for i in range(n, nrows * ncols):
        axes_q[i // ncols][i % ncols].axis("off")
    fig_q.suptitle(f"VEV Vouchers PnL — Submission {subnum}", fontsize=13)
    fig_q.tight_layout()
    out = dd / f"{subnum}_vouchers_pnl.png"
    fig_q.savefig(out, dpi=150)
    saved_paths.append(out)

for p in saved_paths:
    print(f"saved {p}")
plt.show()
