import json
import csv
import io
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # no GUI display
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
# Round 5 categories: one PnL figure per category, 5 product curves overlaid
# ---------------------------------------------------------------------------
CATEGORIES = {
    "Galaxy_Sounds_Recorders": [
        "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "Vertical_Sleeping_Pods": [
        "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    ],
    "Organic_Microchips": [
        "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    ],
    "Purification_Pebbles": [
        "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    ],
    "Domestic_Robots": [
        "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
        "ROBOT_LAUNDRY", "ROBOT_IRONING",
    ],
    "UV_Visors": [
        "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
        "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    ],
    "Instant_Translators": [
        "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "Construction_Panels": [
        "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    ],
    "Liquid_Breath_Oxygen_Shakes": [
        "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    ],
    "Protein_Snack_Packs": [
        "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
    ],
}

saved_paths = []


def plot_product_panel(ax, product):
    """Price + buy/sell marks (left axis), PnL curve (right axis)."""
    plot_price(ax, product, annotate_qty=True)
    final_pnl = 0.0
    if product in pnl:
        xs, ys = pnl[product]
        ax_r = ax.twinx()
        ax_r.plot(xs, ys, color="purple", linewidth=1.1,
                  alpha=0.85, label="PnL", zorder=5)
        ax_r.set_ylabel("P&L", color="purple")
        ax_r.tick_params(axis="y", labelcolor="purple")
        if ys:
            final_pnl = ys[-1]
    ax.set_title(f"{product}  (PnL {final_pnl:+.0f})", fontsize=10)
    return final_pnl


for cat_name, products in CATEGORIES.items():
    present = [p for p in products if p in pnl]
    if not present:
        continue

    ncols = len(products)  # always 5 for R5
    fig, axes = plt.subplots(1, ncols, figsize=(4.6 * ncols, 4.2),
                             squeeze=False)
    finals = {}
    for i, p in enumerate(products):
        ax = axes[0][i]
        if p in pnl or p in prices or p in mid:
            finals[p] = plot_product_panel(ax, p)
        else:
            ax.set_title(f"{p}  (no data)", fontsize=10)
            ax.axis("off")

    total = sum(finals.values())
    fig.suptitle(f"{cat_name.replace('_', ' ')} — Submission {subnum}  "
                 f"(Category PnL: {total:+.0f})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out = dd / f"{subnum}_{cat_name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(out)

for p in saved_paths:
    print(f"saved {p}")
