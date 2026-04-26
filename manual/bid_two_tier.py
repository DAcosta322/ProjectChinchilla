"""
Interactive simulator for the two-bid sealed auction manual problem.

Setup:
  - Counterparty reserve prices are uniform on {670, 675, ..., 920} (51 values).
  - Submit two bids b1, b2 (integers; b1 <= b2 conventionally).
  - Trade rules per counterparty with reserve r:
        * if b1 >= r:  trade at b1, profit = 920 - b1
        * elif b2 >= r:
              - if b2 >  avg_b2:  trade at b2,  profit = 920 - b2
              - if b2 <= avg_b2:  trade at b2 with prob ((920-avg_b2)/(920-b2))**3
                                  expected profit = (920 - b2) * factor
                                                  = (920-avg_b2)**3 / (920-b2)**2
        * else:        no trade
  - Resale on next day at 920.

Run:
    python bid_two_tier.py
"""

from __future__ import annotations

import numpy as np

LOW = 670
HIGH = 920
STEP = 5
FAIR = 920
RESERVES = np.arange(LOW, HIGH + STEP, STEP)  # 670..920 step 5  (length 51)


# ---------- core math ----------

def per_reserve_profit(b1: int, b2: int, avg_b2: float, r: int) -> float:
    """Expected profit from one counterparty with reserve r."""
    if b1 >= r:
        return FAIR - b1
    if b2 >= r:
        if b2 > avg_b2:
            return FAIR - b2
        # b2 <= avg_b2 -> probability penalty
        if b2 >= FAIR:
            return 0.0
        factor = ((FAIR - avg_b2) / (FAIR - b2)) ** 3
        return (FAIR - b2) * factor
    return 0.0


def expected_profit(b1: int, b2: int, avg_b2: float,
                    reserves: np.ndarray = RESERVES) -> float:
    """Average profit across the uniform reserve distribution."""
    total = 0.0
    for r in reserves:
        total += per_reserve_profit(b1, b2, avg_b2, int(r))
    return total / len(reserves)


def grid_search(avg_b2: float,
                lo: int = LOW, hi: int = HIGH,
                step: int = 1) -> tuple[float, int, int, np.ndarray]:
    """Brute force over integer (b1, b2) grid. Returns (best_ev, b1, b2, ev_matrix)."""
    bids = np.arange(lo, hi + 1, step)
    M = np.full((len(bids), len(bids)), -np.inf)
    best = (-np.inf, lo, lo)
    for i, b1 in enumerate(bids):
        for j, b2 in enumerate(bids):
            if b2 < b1:
                continue
            ev = expected_profit(int(b1), int(b2), avg_b2)
            M[i, j] = ev
            if ev > best[0]:
                best = (ev, int(b1), int(b2))
    return best[0], best[1], best[2], M


# ---------- non-uniform reserve priors ----------

def expected_profit_pmf(b1: int, b2: int, avg_b2: float,
                        probs: np.ndarray,
                        reserves: np.ndarray = RESERVES) -> float:
    total = 0.0
    for r, p in zip(reserves, probs):
        total += p * per_reserve_profit(b1, b2, avg_b2, int(r))
    return total


def grid_search_pmf(avg_b2: float, probs: np.ndarray,
                    reserves: np.ndarray = RESERVES,
                    lo: int = LOW, hi: int = HIGH,
                    step: int = 1) -> tuple[float, int, int, np.ndarray]:
    bids = np.arange(lo, hi + 1, step)
    M = np.full((len(bids), len(bids)), -np.inf)
    best = (-np.inf, lo, lo)
    for i, b1 in enumerate(bids):
        for j, b2 in enumerate(bids):
            if b2 < b1:
                continue
            ev = expected_profit_pmf(int(b1), int(b2), avg_b2, probs, reserves)
            M[i, j] = ev
            if ev > best[0]:
                best = (ev, int(b1), int(b2))
    return best[0], best[1], best[2], M


def pmf_uniform(reserves: np.ndarray = RESERVES) -> np.ndarray:
    return np.full(len(reserves), 1.0 / len(reserves))


def pmf_normal(mean: float, std: float,
               reserves: np.ndarray = RESERVES) -> np.ndarray:
    p = np.exp(-0.5 * ((reserves - mean) / std) ** 2)
    return p / p.sum()


def pmf_normal_flat(mean: float, std: float, flat_weight: float = 0.3,
                    reserves: np.ndarray = RESERVES) -> np.ndarray:
    """Mixture: (1-w) * Normal + w * Uniform.   w is the 'flat rate'."""
    return (1 - flat_weight) * pmf_normal(mean, std, reserves) \
           + flat_weight * pmf_uniform(reserves)


def pmf_left_skewed(alpha: float = 4.0, beta: float = 2.0,
                    lo: int = LOW, hi: int = HIGH,
                    reserves: np.ndarray = RESERVES) -> np.ndarray:
    """
    Beta(alpha, beta) rescaled to [lo, hi]. alpha > beta -> mean shifted right
    of midpoint, long left tail (negative skew = 'left-skewed').
    """
    x = (reserves - lo) / (hi - lo)
    x = np.clip(x, 1e-9, 1 - 1e-9)
    p = x ** (alpha - 1) * (1 - x) ** (beta - 1)
    return p / p.sum()


def pmf_mean(probs: np.ndarray, reserves: np.ndarray = RESERVES) -> float:
    return float(np.dot(reserves, probs))


def best_response(avg_b2: float, step: int = 1) -> tuple[float, int, int]:
    ev, b1, b2, _ = grid_search(avg_b2, step=step)
    return ev, b1, b2


def bounded_geometric(lo: int, hi: int, q: float,
                      step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Truncated geometric on {lo, lo+step, ..., hi}.
        P(X = lo + k*step) proportional to q**k,  k = 0, 1, ..., n
    q in (0, 1) puts mass at the lower end (smaller q = sharper decay).
    q = 1 collapses to uniform.
    Returns (values, probabilities).
    """
    if not (0.0 < q <= 1.0):
        raise ValueError("q must be in (0, 1]")
    values = np.arange(lo, hi + 1, step)
    n = len(values)
    if abs(q - 1.0) < 1e-12:
        probs = np.full(n, 1.0 / n)
    else:
        weights = q ** np.arange(n)
        probs = weights / weights.sum()
    return values, probs


def bounded_geometric_mean(lo: int, hi: int, q: float, step: int = 1) -> float:
    values, probs = bounded_geometric(lo, hi, q, step=step)
    return float(np.dot(values, probs))


def find_symmetric_equilibrium(verbose: bool = False) -> tuple[float, int, int, float]:
    """
    Symmetric Nash: find avg_b2* s.t. the best-response b2 equals avg_b2*.
    Sweep candidate avg_b2 over the bid grid; pick the largest fixed point
    (highest such anchor; the lower ones are typically unstable).
    Returns (ev_at_eq, b1*, b2*, avg_b2*).
    """
    candidates = []
    for m in range(LOW, HIGH + 1):
        ev, b1, b2 = best_response(float(m))
        if verbose:
            print(f"  avg_b2={m:4d} -> BR (b1={b1}, b2={b2}, EV={ev:.3f})")
        if b2 == m:
            candidates.append((ev, b1, b2, float(m)))
    if not candidates:
        return (np.nan, -1, -1, np.nan)
    # Prefer the highest-EV fixed point
    return max(candidates, key=lambda x: x[0])


# ---------- presentation ----------

def show_breakdown(b1: int, b2: int, avg_b2: float) -> None:
    print()
    print(f"Bids:  b1 = {b1}   b2 = {b2}   assumed avg_b2 = {avg_b2:g}")
    print(f"  reserve | route        | profit")
    print(f"  --------+--------------+-------")
    n_b1 = n_b2_full = n_b2_pen = n_skip = 0
    total = 0.0
    for r in RESERVES:
        r = int(r)
        if b1 >= r:
            route = "b1 fill"
            p = FAIR - b1
            n_b1 += 1
        elif b2 >= r:
            if b2 > avg_b2:
                route = "b2 full"
                p = FAIR - b2
                n_b2_full += 1
            else:
                route = "b2 penalised"
                if b2 >= FAIR:
                    p = 0.0
                else:
                    p = (FAIR - avg_b2) ** 3 / (FAIR - b2) ** 2
                n_b2_pen += 1
        else:
            route = "no trade"
            p = 0.0
            n_skip += 1
        total += p
        print(f"  {r:7d} | {route:12s} | {p:6.2f}")
    avg = total / len(RESERVES)
    print(f"  --------+--------------+-------")
    print(f"  Counts: b1={n_b1}  b2_full={n_b2_full}  b2_pen={n_b2_pen}  miss={n_skip}")
    print(f"  Total expected profit per counterparty: {avg:.4f}")


def plot_heatmap(avg_b2: float, step: int = 1) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return
    _, _, _, M = grid_search(avg_b2, step=step)
    bids = np.arange(LOW, HIGH + 1, step)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(M, origin="lower",
                   extent=[bids[0], bids[-1], bids[0], bids[-1]],
                   aspect="auto", cmap="viridis")
    plt.colorbar(im, ax=ax, label="E[profit per counterparty]")
    iy, ix = np.unravel_index(np.nanargmax(M), M.shape)
    ax.scatter([bids[ix]], [bids[iy]], color="red", marker="x", s=80,
               label=f"best b1={bids[iy]}, b2={bids[ix]}, EV={M[iy, ix]:.3f}")
    ax.set_xlabel("b2")
    ax.set_ylabel("b1")
    ax.set_title(f"Expected profit heatmap (avg_b2 = {avg_b2:g})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_best_response_curve(step: int = 1) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping plot.")
        return
    avgs = np.arange(LOW, HIGH + 1, step)
    br_b2 = np.empty_like(avgs, dtype=float)
    br_b1 = np.empty_like(avgs, dtype=float)
    evs = np.empty_like(avgs, dtype=float)
    for i, m in enumerate(avgs):
        ev, b1, b2 = best_response(float(m))
        br_b1[i], br_b2[i], evs[i] = b1, b2, ev
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    ax1.plot(avgs, avgs, "k--", lw=1, label="b2 = avg_b2 (45 degree)")
    ax1.plot(avgs, br_b2, label="best-response b2")
    ax1.plot(avgs, br_b1, label="best-response b1")
    fp_mask = (br_b2 == avgs)
    if fp_mask.any():
        ax1.scatter(avgs[fp_mask], br_b2[fp_mask], color="red", s=60, zorder=5,
                    label="symmetric equilibria")
    ax1.set_ylabel("bid")
    ax1.set_title("Best-response curve")
    ax1.legend()
    ax1.grid(alpha=0.3)
    ax2.plot(avgs, evs, color="purple")
    ax2.set_xlabel("market avg_b2")
    ax2.set_ylabel("EV at best response")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# ---------- interactive menu ----------

def geometric_avg_and_br(lo: int = 835, hi: int = 880,
                         q_values: tuple[float, ...] = (0.30, 0.50, 0.70,
                                                        0.85, 0.95, 1.00),
                         step: int = 1) -> None:
    """
    Model the field's b2 as a truncated geometric on {lo..hi} with decay q.
    Print E[b2], the resulting best-response (b1*, b2*), and EV.
    """
    print(f"\n  field b2 ~ truncated geometric on [{lo}, {hi}] step {step}")
    print(f"  P(X = {lo}+k*{step}) ~ q**k   (smaller q = more mass on {lo})")
    print()
    print(f"  {'q':>5} | {'E[b2]':>8} | {'BR b1':>5} | {'BR b2':>5} | "
          f"{'EV':>7} | {'P(>={lo})':>9}")
    print("  " + "-" * 60)
    for q in q_values:
        m = bounded_geometric_mean(lo, hi, q, step=step)
        ev, b1, b2 = best_response(m)
        # how peaked at lo
        _, probs = bounded_geometric(lo, hi, q, step=step)
        p0 = probs[0]
        print(f"  {q:5.2f} | {m:8.3f} | {b1:5d} | {b2:5d} | {ev:7.3f} | {p0:9.3f}")


MENU = """
============================================================
 Two-bid auction simulator (reserves 670..920 step 5)
------------------------------------------------------------
  1) Score a (b1, b2) pair vs assumed avg_b2
  2) Find best response (b1, b2) for given avg_b2
  3) Find symmetric Nash equilibrium (b2 = avg_b2)
  4) Plot EV heatmap over (b1, b2) for given avg_b2
  5) Plot best-response curve over avg_b2 in [670, 920]
  6) Field b2 ~ bounded geometric on [835, 880] -> mean + BR
  q) quit
============================================================
"""


def _ask_int(prompt: str, default: int | None = None) -> int:
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            return int(s)
        except ValueError:
            print("  please enter an integer")


def _ask_float(prompt: str, default: float | None = None) -> float:
    while True:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        try:
            return float(s)
        except ValueError:
            print("  please enter a number")


def main() -> None:
    while True:
        print(MENU)
        choice = input("choose: ").strip().lower()
        if choice in ("q", "quit", "exit"):
            return
        try:
            if choice == "1":
                b1 = _ask_int("  b1 [780]: ", 780)
                b2 = _ask_int("  b2 [840]: ", 840)
                m  = _ask_float("  avg_b2 [840]: ", 840.0)
                show_breakdown(b1, b2, m)
            elif choice == "2":
                m = _ask_float("  avg_b2 [840]: ", 840.0)
                ev, b1, b2 = best_response(m)
                print(f"\n  best response: b1={b1}, b2={b2}, EV={ev:.4f}")
                show_breakdown(b1, b2, m)
            elif choice == "3":
                print("  searching symmetric equilibria...")
                ev, b1, b2, m = find_symmetric_equilibrium()
                if np.isnan(ev):
                    print("  no symmetric equilibrium found on integer grid")
                else:
                    print(f"\n  symmetric equilibrium: avg_b2*={m:g}, "
                          f"b1*={b1}, b2*={b2}, EV={ev:.4f}")
                    show_breakdown(b1, b2, m)
            elif choice == "4":
                m = _ask_float("  avg_b2 [840]: ", 840.0)
                plot_heatmap(m)
            elif choice == "5":
                plot_best_response_curve()
            elif choice == "6":
                lo = _ask_int("  geometric lo [835]: ", 835)
                hi = _ask_int("  geometric hi [880]: ", 880)
                step = _ask_int("  step (1 or 5) [1]: ", 1)
                qs_in = input("  q values (comma sep) [0.3,0.5,0.7,0.85,0.95,1.0]: ").strip()
                if qs_in:
                    qs = tuple(float(x) for x in qs_in.split(","))
                else:
                    qs = (0.30, 0.50, 0.70, 0.85, 0.95, 1.00)
                geometric_avg_and_br(lo, hi, qs, step=step)
                # also let user pick a single q and see breakdown
                pick = input("\n  drill into one q? [blank to skip]: ").strip()
                if pick:
                    q = float(pick)
                    m = bounded_geometric_mean(lo, hi, q, step=step)
                    print(f"\n  E[b2] = {m:.3f}  ->  best response:")
                    ev, b1, b2 = best_response(m)
                    print(f"    b1={b1}, b2={b2}, EV={ev:.4f}")
                    show_breakdown(b1, b2, m)
            else:
                print("  unknown choice")
        except (KeyboardInterrupt, EOFError):
            print()
            return


if __name__ == "__main__":
    main()
