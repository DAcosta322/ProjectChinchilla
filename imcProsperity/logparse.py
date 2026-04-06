import json
import csv
import io
from pathlib import Path
import matplotlib.pyplot as plt

subnum = input("Subnum:")

dd = "Dump/" + subnum
jj = subnum + ".json"
ll = subnum + ".log"
d = Path(dd)
j = json.loads((d / jj).read_text())
log = json.loads((d / ll).read_text())

print("profit", j["profit"])
print("positions", j["positions"])
print("activities rows", len(j["activitiesLog"].splitlines()))

# trade history stats
trade = log["tradeHistory"]
print("trades", len(trade))
print("symbols", {s: sum(1 for t in trade if t["symbol"] == s) for s in {"TOMATOES", "EMERALDS"}})

# parse activitiesLog CSV into per-product PNL series
reader = csv.DictReader(io.StringIO(j["activitiesLog"]), delimiter=";")
pnl = {}  # product -> (timestamps[], pnl[])
days_seen = set()
for row in reader:
    product = row["product"]
    day = int(row["day"])
    ts = int(row["timestamp"])
    days_seen.add(day)
    # use sequential index: day_index * 1M + timestamp
    x = ts
    y = float(row["profit_and_loss"])
    if product not in pnl:
        pnl[product] = ([], [])
    pnl[product][0].append(x)
    pnl[product][1].append(y)

# plot separate PNL graphs
products = sorted(pnl.keys())
fig, axes = plt.subplots(len(products), 1, figsize=(12, 5 * len(products)), sharex=True)
if len(products) == 1:
    axes = [axes]

for ax, product in zip(axes, products):
    xs, ys = pnl[product]
    ax.plot(xs, ys, linewidth=1)
    ax.set_title(f"{product} PNL")
    ax.set_ylabel("Profit & Loss")
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Timestamp")
fig.suptitle(f"PNL by Product — Submission {subnum}", fontsize=14)
fig.tight_layout()

out = d / f"{subnum}_pnl.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")

# parse trade history into buy/sell/self price series per product
prices = {}  # product -> {"buy": (ts[], px[]), "sell": (ts[], px[]), "self": (ts[], px[])}
for t in trade:
    sym = t["symbol"]
    if sym not in prices:
        prices[sym] = {"buy": ([], []), "sell": ([], []), "self": ([], [])}
    ts = t["timestamp"]
    px = t["price"]
    is_buyer = t["buyer"] == "SUBMISSION"
    is_seller = t["seller"] == "SUBMISSION"
    if is_buyer and is_seller:
        prices[sym]["self"][0].append(ts)
        prices[sym]["self"][1].append(px)
    elif is_buyer:
        prices[sym]["buy"][0].append(ts)
        prices[sym]["buy"][1].append(px)
    elif is_seller:
        prices[sym]["sell"][0].append(ts)
        prices[sym]["sell"][1].append(px)

# plot price graphs
trade_products = sorted(prices.keys())
fig2, axes2 = plt.subplots(len(trade_products), 1, figsize=(12, 5 * len(trade_products)), sharex=True)
if len(trade_products) == 1:
    axes2 = [axes2]

for ax, product in zip(axes2, trade_products):
    for label, color, marker in [("buy", "green", "^"), ("sell", "red", "v"), ("self", "blue", "o")]:
        xs, ys = prices[product][label]
        if xs:
            ax.scatter(xs, ys, c=color, marker=marker, label=label, alpha=0.7, s=20)
    ax.set_title(f"{product} Trade Prices")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)

axes2[-1].set_xlabel("Timestamp")
fig2.suptitle(f"Trade Prices — Submission {subnum}", fontsize=14)
fig2.tight_layout()

out2 = d / f"{subnum}_prices.png"
fig2.savefig(out2, dpi=150)
print(f"saved {out2}")
plt.show()
