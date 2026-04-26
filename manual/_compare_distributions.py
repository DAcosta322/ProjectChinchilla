"""Quick comparison of (b1, b2) best response across four reserve-price priors,
all evaluated at avg_b2 = 846."""
import numpy as np
from manual.bid_two_tier import (
    RESERVES, pmf_uniform, pmf_normal, pmf_normal_flat, pmf_left_skewed,
    pmf_mean, grid_search_pmf, expected_profit_pmf,
)

AVG = 846

scenarios = [
    ("1. Normal(795, 50) on [670,920]",        pmf_normal(795, 50)),
    ("2. Uniform on {670..920 step 5}",        pmf_uniform()),
    ("3. Normal(795,50) + 30% flat (mixture)", pmf_normal_flat(795, 50, 0.30)),
    ("4. Left-skewed Beta(4,2) on [670,920]",  pmf_left_skewed(4.0, 2.0)),
]

bar = "=" * 78
print(bar)
print(f"  Best response under avg_b2 = {AVG}, grid search step=1 over [670, 920]")
print(bar)
print()
print(f"  {'scenario':45s} {'E[r]':>7}  {'BR (b1, b2)':>13}  {'EV':>8}")
print("  " + "-" * 76)

results = []
for name, probs in scenarios:
    mean_r = pmf_mean(probs)
    ev, b1, b2, M = grid_search_pmf(AVG, probs)
    pair = f"({b1}, {b2})"
    print(f"  {name:45s} {mean_r:7.2f}  {pair:>13s}  {ev:8.3f}")
    results.append((name, probs, b1, b2, ev, M))

print()
print("Top-5 (b1, b2) by EV per scenario:")
print()
bids = np.arange(670, 921)
for name, probs, _, _, _, M in results:
    flat = []
    for i, B1 in enumerate(bids):
        for j, B2 in enumerate(bids):
            if B2 < B1:
                continue
            flat.append((M[i, j], int(B1), int(B2)))
    flat.sort(reverse=True)
    print(f"  {name}")
    for k in range(5):
        e, a, b = flat[k]
        print(f"    {k+1}. (b1={a}, b2={b})  EV={e:.3f}")
    print()

print("Cross-evaluation matrix (row = bid pair, col = actual reserve scenario)")
print()
short = ["Norm", "Unif", "N+flat", "LeftSk"]
header = "  " + " " * 14 + "  ".join(f"{s:>10}" for s in short) + "    BR-of"
print(header)
print("  " + "-" * 76)
for name_row, _, b1, b2, _, _ in results:
    cells = []
    for _, probs, _, _, _, _ in results:
        e = expected_profit_pmf(b1, b2, AVG, probs)
        cells.append(f"{e:10.3f}")
    pair = f"({b1:3d}, {b2:3d})"
    print(f"  {pair:>14s}  " + "  ".join(cells) + f"    {name_row[:6]}")

print()
print("Worst-case across the four scenarios for each candidate pair:")
candidates = [(b1, b2) for _, _, b1, b2, _, _ in results]
# add a few common reference pairs
candidates += [(750, 840), (760, 860), (753, 837), (750, 846), (760, 846)]
seen = set()
print(f"  {'pair':>14s}  {'worst':>8s}  {'min over scenarios':s}")
for b1, b2 in candidates:
    if (b1, b2) in seen:
        continue
    seen.add((b1, b2))
    evs = [expected_profit_pmf(b1, b2, AVG, p) for _, p, _, _, _, _ in results]
    worst = min(evs)
    spread = ", ".join(f"{e:.2f}" for e in evs)
    print(f"  ({b1:3d}, {b2:3d})  {worst:8.3f}  [{spread}]")
