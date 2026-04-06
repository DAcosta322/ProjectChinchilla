import json
import csv
import io
from pathlib import Path
import matplotlib.pyplot as plt

subnum = "53755"

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
plt.show()
