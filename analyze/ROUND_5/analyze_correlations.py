"""Round 5 correlation analysis.

Produces:
  data/ROUND_5/correlations/
    cat_<NAME>.png       — 5x5 intra-category return correlation, per day + averaged
    master_50x50.png     — 50x50 return correlation, 3-day averaged, ordered by category
    drift_summary.csv    — per-product per-day drift (end - start)
    drift_consistency.png — scatter of day-N drift vs day-M drift across all 50 products
    summary.txt          — text report

Run: python analyze_correlations.py
"""

import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ROUND_5"
OUT_DIR = DATA_DIR / "correlations"
DAYS = [2, 3, 4]

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
ORDERED = [p for members in CATEGORIES.values() for p in members]
PROD2IDX = {p: i for i, p in enumerate(ORDERED)}


def read_mids(day):
    """Return {product: dense np.array of mid_price by tick}, sorted timestamps."""
    path = DATA_DIR / f"prices_round_5_day_{day}.csv"
    rows = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            try:
                m = float(row["mid_price"])
            except (ValueError, TypeError):
                continue
            ts = int(row["timestamp"])
            rows.setdefault(row["product"], {})[ts] = m
    all_ts = sorted({t for v in rows.values() for t in v})
    out = {}
    for p, mp in rows.items():
        out[p] = np.array([mp.get(t, np.nan) for t in all_ts])
    return out, all_ts


def returns_matrix(mids):
    """(T-1, 50) first-differences of mid, ordered by ORDERED. NaNs forward-filled."""
    cols = []
    for p in ORDERED:
        arr = mids[p].astype(float)
        # Forward fill any NaN
        mask = np.isnan(arr)
        if mask.any():
            idx = np.where(~mask, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            arr = arr[idx]
        cols.append(np.diff(arr))
    return np.stack(cols, axis=1)


def safe_corr(M):
    """Return correlation matrix robust to zero-variance columns."""
    std = M.std(axis=0)
    safe = np.where(std == 0, 1.0, std)
    Z = (M - M.mean(axis=0)) / safe
    C = (Z.T @ Z) / len(M)
    C[std == 0, :] = 0
    C[:, std == 0] = 0
    np.fill_diagonal(C, 1.0)
    return C


def heatmap(ax, mat, labels, title, vmin=-1, vmax=1):
    im = ax.imshow(mat, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = mat[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(v) > 0.5 else "black")
    return im


def plot_category(name, members, per_day_C):
    short = [m.replace(name + "_", "") for m in members]
    idx = [PROD2IDX[m] for m in members]

    sub = {d: per_day_C[d][np.ix_(idx, idx)] for d in DAYS}
    avg = np.mean([sub[d] for d in DAYS], axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, d in zip(axes[:3], DAYS):
        heatmap(ax, sub[d], short, f"{name} day {d}")
    im = heatmap(axes[3], avg, short, f"{name} avg of {len(DAYS)} days")
    fig.suptitle(f"{name} — intra-category return correlation", fontsize=12)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02)
    fig.savefig(OUT_DIR / f"cat_{name}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_master(avg_C):
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(avg_C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(50))
    ax.set_yticks(range(50))
    short_labels = []
    for p in ORDERED:
        for cat in CATEGORIES:
            if p.startswith(cat + "_"):
                short_labels.append(f"{cat[:4]}_{p[len(cat)+1:]}"[:14])
                break
    ax.set_xticklabels(short_labels, rotation=90, fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)

    # Draw category boundary lines
    pos = 0
    for cat, members in CATEGORIES.items():
        pos += len(members)
        if pos < 50:
            ax.axhline(pos - 0.5, color="black", lw=0.6)
            ax.axvline(pos - 0.5, color="black", lw=0.6)

    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(f"50x50 return correlation, mean of days {DAYS}", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "master_50x50.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT_DIR.mkdir(exist_ok=True)

    per_day_R = {}
    per_day_C = {}
    per_day_mids = {}
    for d in DAYS:
        print(f"Reading day {d} ...")
        mids, ts = read_mids(d)
        per_day_mids[d] = mids
        R = returns_matrix(mids)
        per_day_R[d] = R
        per_day_C[d] = safe_corr(R)
        print(f"  ticks={len(ts)}, returns shape={R.shape}")

    avg_C = np.mean([per_day_C[d] for d in DAYS], axis=0)

    print("\nPlotting per-category panels ...")
    for cat, members in CATEGORIES.items():
        plot_category(cat, members, per_day_C)
    print("Plotting master 50x50 ...")
    plot_master(avg_C)

    # ---- Drift summary ----
    drift = {p: {} for p in ORDERED}
    for d in DAYS:
        for p in ORDERED:
            arr = per_day_mids[d][p]
            drift[p][d] = float(arr[-1] - arr[0])

    csv_path = OUT_DIR / "drift_summary.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["product"] + [f"day_{d}_drift" for d in DAYS] +
                   ["mean_drift", "drift_sign_consistency"])
        for p in ORDERED:
            ds = [drift[p][d] for d in DAYS]
            sign_cons = sum(1 for x in ds if x > 0) / len(ds) if any(d > 0 for d in ds) else 0
            w.writerow([p] + [f"{x:.1f}" for x in ds] +
                       [f"{np.mean(ds):.1f}",
                        f"{max(sum(1 for x in ds if x > 0), sum(1 for x in ds if x < 0)) / len(ds):.2f}"])

    # Drift consistency scatter
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pairs = [(2, 3), (2, 4), (3, 4)]
    cat_colors = plt.cm.tab10(np.linspace(0, 1, len(CATEGORIES)))
    cat_idx = {}
    for i, cat in enumerate(CATEGORIES):
        for m in CATEGORIES[cat]:
            cat_idx[m] = i

    for ax, (a, b) in zip(axes, pairs):
        xs = [drift[p][a] for p in ORDERED]
        ys = [drift[p][b] for p in ORDERED]
        cs = [cat_colors[cat_idx[p]] for p in ORDERED]
        ax.scatter(xs, ys, c=cs, s=40, alpha=0.85, edgecolors="black", linewidths=0.4)
        # Per-product label only for outliers
        threshold = max(np.percentile(np.abs(xs + ys), 75), 500)
        for x, y, p in zip(xs, ys, ORDERED):
            if abs(x) > threshold or abs(y) > threshold:
                ax.annotate(p[-12:], (x, y), fontsize=6, alpha=0.7)
        c = np.corrcoef(xs, ys)[0, 1]
        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"day {a} drift")
        ax.set_ylabel(f"day {b} drift")
        ax.set_title(f"day {a} vs day {b}, ρ={c:.3f}")
        ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([], [], marker="o", linestyle="", color=cat_colors[i],
                          label=cat, markersize=8)
               for i, cat in enumerate(CATEGORIES)]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=9, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Cross-day drift consistency (each point = one product)", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(OUT_DIR / "drift_consistency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- Text summary ----
    with open(OUT_DIR / "summary.txt", "w") as f:
        f.write("=== Round 5 Correlation Summary ===\n\n")

        f.write("--- Within-category mean off-diagonal correlation (avg of days) ---\n")
        for cat, members in CATEGORIES.items():
            idx = [PROD2IDX[m] for m in members]
            sub = avg_C[np.ix_(idx, idx)]
            off_diag = sub[~np.eye(len(idx), dtype=bool)]
            f.write(f"  {cat:14s}  mean={off_diag.mean():+.3f}  "
                    f"min={off_diag.min():+.3f}  max={off_diag.max():+.3f}\n")

        f.write("\n--- Within vs between category (day-averaged) ---\n")
        in_mask = np.zeros((50, 50), dtype=bool)
        pos = 0
        for members in CATEGORIES.values():
            n = len(members)
            in_mask[pos:pos+n, pos:pos+n] = True
            pos += n
        np.fill_diagonal(in_mask, False)
        within = avg_C[in_mask]
        between = avg_C[~in_mask & ~np.eye(50, dtype=bool)]
        f.write(f"  within-category mean ρ:  {within.mean():+.4f}  (n={within.size})\n")
        f.write(f"  between-category mean ρ: {between.mean():+.4f}  (n={between.size})\n")

        f.write("\n--- Day-vs-day correlation of per-product drift vector ---\n")
        for a, b in [(2, 3), (2, 4), (3, 4)]:
            xs = [drift[p][a] for p in ORDERED]
            ys = [drift[p][b] for p in ORDERED]
            c = np.corrcoef(xs, ys)[0, 1]
            f.write(f"  day {a} drift vs day {b} drift:  ρ={c:+.4f}\n")

        f.write("\n--- Top 10 within-category pairs by avg correlation ---\n")
        all_pairs = []
        for cat, members in CATEGORIES.items():
            idx = [PROD2IDX[m] for m in members]
            for i in range(len(idx)):
                for j in range(i+1, len(idx)):
                    all_pairs.append((avg_C[idx[i], idx[j]], members[i], members[j], cat))
        all_pairs.sort(reverse=True)
        for c, a, b, cat in all_pairs[:10]:
            f.write(f"  {c:+.3f}  [{cat}]  {a}  /  {b}\n")

        f.write("\n--- Bottom 10 within-category pairs ---\n")
        for c, a, b, cat in all_pairs[-10:]:
            f.write(f"  {c:+.3f}  [{cat}]  {a}  /  {b}\n")

        f.write("\n--- Strongest cross-category pairs (|ρ|, top 15) ---\n")
        cross = []
        pos = 0
        cat_of = {}
        for cat, members in CATEGORIES.items():
            for m in members:
                cat_of[m] = cat
        for i in range(50):
            for j in range(i+1, 50):
                if cat_of[ORDERED[i]] != cat_of[ORDERED[j]]:
                    cross.append((avg_C[i, j], ORDERED[i], ORDERED[j]))
        cross.sort(key=lambda x: -abs(x[0]))
        for c, a, b in cross[:15]:
            f.write(f"  {c:+.3f}  {a}  /  {b}\n")

    print(f"\nDone. Outputs in {OUT_DIR}")
    with open(OUT_DIR / "summary.txt") as f:
        print("\n" + f.read())


if __name__ == "__main__":
    main()
