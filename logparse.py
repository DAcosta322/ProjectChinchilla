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

# combined plot: PNL (top) + Trade Prices (bottom) per product, one figure
products = sorted(pnl.keys())
fig, axes = plt.subplots(len(products), 2, figsize=(16, 5 * len(products)),
                         gridspec_kw={"width_ratios": [1, 1]})
if len(products) == 1:
    axes = [axes]

for row_axes, product in zip(axes, products):
    ax_pnl, ax_price = row_axes

    # PNL subplot
    xs, ys = pnl[product]
    ax_pnl.plot(xs, ys, linewidth=1)
    ax_pnl.set_title(f"{product} PNL")
    ax_pnl.set_ylabel("Profit & Loss")
    ax_pnl.set_xlabel("Timestamp")
    ax_pnl.grid(True, alpha=0.3)

    # Trade prices subplot — mid price curve
    if product in mid:
        mxs, mys = mid[product]
        ax_price.plot(mxs, mys, color="gray", linewidth=0.8, alpha=0.6, label="mid", zorder=1)

    # Trade prices subplot — scatter
    if product in prices:
        for label, color, marker in [("buy", "green", "^"), ("sell", "red", "v"), ("self", "blue", "o")]:
            txs, tys, tqs = prices[product][label]
            if txs:
                # Scale alpha by quantity: map [min_qty, max_qty] -> [0.2, 1.0]
                max_q = max(tqs)
                min_q = min(tqs)
                if max_q == min_q:
                    alphas = [0.7] * len(tqs)
                else:
                    alphas = [min(1.0, 0.2 + 0.8 * (q - min_q) / (max_q - min_q)) for q in tqs]
                ax_price.scatter(txs, tys, c=color, marker=marker, label=label, alpha=alphas, s=20)
                # Annotate each point with its quantity
                for x, y, q in zip(txs, tys, tqs):
                    ax_price.annotate(str(q), (x, y), textcoords="offset points",
                                      xytext=(0, 6 if marker == "^" else -10),
                                      fontsize=6, ha="center", color=color, alpha=0.8)
        ax_price.legend()
    ax_price.set_title(f"{product} Trade Prices")
    ax_price.set_ylabel("Price")
    ax_price.set_xlabel("Timestamp")
    ax_price.grid(True, alpha=0.3)

fig.suptitle(f"Submission {subnum}", fontsize=14)
fig.tight_layout()

out = dd / f"{subnum}_combined.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
plt.show()
