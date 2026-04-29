"""Plot mid-price overlays for Round 5 product categories.

One figure per category, all 5 products overlaid on the same axes.
Every product has POS_LIMIT=10.

Usage:
    python plot_categories.py            # prompts for day
    python plot_categories.py 2          # day 2
    python plot_categories.py 2 3 4      # days 2, 3, 4 in sequence
"""

import csv
import sys
from pathlib import Path
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"

CATEGORIES = {
    "GALAXY_SOUNDS": [
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
    ],
    "SLEEP_POD": [
        "SLEEP_POD_SUEDE",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_COTTON",
    ],
    "MICROCHIP": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_SQUARE",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_TRIANGLE",
    ],
    "PEBBLES": [
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    ],
    "ROBOT": [
        "ROBOT_VACUUMING",
        "ROBOT_MOPPING",
        "ROBOT_DISHES",
        "ROBOT_LAUNDRY",
        "ROBOT_IRONING",
    ],
    "UV_VISOR": [
        "UV_VISOR_YELLOW",
        "UV_VISOR_AMBER",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_MAGENTA",
    ],
    "TRANSLATOR": [
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
    ],
    "PANEL": [
        "PANEL_1X2",
        "PANEL_2X2",
        "PANEL_1X4",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "OXYGEN_SHAKE": [
        "OXYGEN_SHAKE_MORNING_BREATH",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_GARLIC",
    ],
    "SNACKPACK": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_RASPBERRY",
    ],
}

PALETTE = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]


def read_prices(path: Path):
    products = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            p = row["product"]
            ts = int(row["timestamp"])
            mid = row["mid_price"]
            try:
                mid = float(mid)
            except (ValueError, TypeError):
                continue
            products.setdefault(p, ([], []))
            products[p][0].append(ts)
            products[p][1].append(mid)
    return products


def plot_category(name, members, mids, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.suptitle(f"{name} — mid prices", fontsize=14)

    for i, prod in enumerate(members):
        if prod not in mids:
            print(f"  WARN: {prod} missing from prices CSV")
            continue
        ts, mid = mids[prod]
        ax.plot(ts, mid, color=PALETTE[i % len(PALETTE)], linewidth=0.7,
                label=prod.replace(name + "_", ""), alpha=0.85)

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Mid price")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = out_dir / f"_CATEGORY_{name}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  saved {out_path.name}")


def run_day(day: int):
    prices_file = DATA_DIR / f"prices_round_5_day_{day}.csv"
    if not prices_file.exists():
        print(f"Error: {prices_file} does not exist")
        return

    print(f"Reading {prices_file.name} ...")
    mids = read_prices(prices_file)
    print(f"  {len(mids)} products loaded")

    out_dir = DATA_DIR / f"day_{day}"
    out_dir.mkdir(exist_ok=True)

    for name, members in CATEGORIES.items():
        print(f"Plotting {name} ...")
        plot_category(name, members, mids, out_dir)

    print(f"\nAll category plots saved to {out_dir}")


def main():
    if len(sys.argv) > 1:
        days = [int(d) for d in sys.argv[1:]]
    else:
        d = input("Day number (2/3/4 or 'all'): ").strip()
        days = [2, 3, 4] if d == "all" else [int(d)]

    for d in days:
        print(f"\n=== Day {d} ===")
        run_day(d)


if __name__ == "__main__":
    main()
