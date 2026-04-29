"""Slice each BT day's PnL series into 100K-tick fragments and analyze.

Reveals length-sensitivity: an algo with high full-day PnL but many losing
100K fragments is fragile to platform replays.

Usage:
    python analyze_fragments.py [--algo ROUND_5/round_5_pebbles_r4]
                                [--round 5] [--products PEBBLES_*]
                                [--fragment-ticks 100000]
"""

import argparse
import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))


def fragment_pnls(pnl_at_ts, frag_ticks):
    """Return list of (start_ts, end_ts, fragment_pnl) per fragment."""
    if not pnl_at_ts:
        return []
    tss = sorted(pnl_at_ts.keys())
    start_ts = tss[0]
    end_ts = tss[-1]
    out = []
    cur = start_ts
    while cur <= end_ts:
        nxt = cur + frag_ticks
        # PnL at start of fragment (or 0 if first)
        if cur == start_ts:
            start_pnl = 0.0
        else:
            # find largest ts < cur
            prior = max((t for t in tss if t < cur), default=None)
            start_pnl = pnl_at_ts[prior] if prior is not None else 0.0
        # PnL at end of fragment
        last_in_frag = max((t for t in tss if t < nxt), default=None)
        if last_in_frag is None:
            break
        end_pnl = pnl_at_ts[last_in_frag]
        out.append((cur, last_in_frag, end_pnl - start_pnl))
        cur = nxt
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", default="ROUND_5/round_5_pebbles_r4")
    ap.add_argument("--round", type=int, default=5)
    ap.add_argument("--days", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--fragment-ticks", type=int, default=100000)
    args = ap.parse_args()

    algo_path = REPO_ROOT / "algorithms" / f"{args.algo}.py"
    spec = importlib.util.spec_from_file_location("trader_algo", algo_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    import backtester as BT
    reader = BT.DataReader(REPO_ROOT / "data")

    print(f"algo: {args.algo}  round: {args.round}  fragment: {args.fragment_ticks} ticks")
    print()

    all_frags = []
    for d in args.days:
        with contextlib.redirect_stdout(io.StringIO()):
            r = BT.run_backtest(module, reader, args.round, d)
        if not r:
            continue
        frags = fragment_pnls(r["pnl_at_ts"], args.fragment_ticks)
        print(f"DAY {d}  full-day PnL: {r['profit']:>10,.0f}  fragments: {len(frags)}")
        for i, (s, e, p) in enumerate(frags):
            tag = "+" if p >= 0 else "-"
            print(f"  frag {i:2d} [{s:>7}..{e:>7}]  PnL: {p:>+10,.0f}  {tag}")
        # Per-product per-fragment
        print(f"  per-product:")
        prod_frags = {}
        for prod, ts_pnl in r["pnl_by_prod_at_ts"].items():
            if not ts_pnl: continue
            pf = fragment_pnls(ts_pnl, args.fragment_ticks)
            if any(abs(x[2]) > 1 for x in pf):
                prod_frags[prod] = [x[2] for x in pf]
        for prod, vals in sorted(prod_frags.items()):
            row = " ".join(f"{v:>+8.0f}" for v in vals)
            print(f"    {prod:14}  {row}   tot={sum(vals):>+9.0f}")
        all_frags.extend(frags)
        print()

    if all_frags:
        pnls = [p for _, _, p in all_frags]
        print(f"=== Summary across all {len(pnls)} fragments ===")
        print(f"  total: {sum(pnls):>+10,.0f}")
        print(f"  min:   {min(pnls):>+10,.0f}")
        print(f"  max:   {max(pnls):>+10,.0f}")
        print(f"  mean:  {sum(pnls)/len(pnls):>+10,.0f}")
        pos = sum(1 for p in pnls if p > 0)
        neg = sum(1 for p in pnls if p < 0)
        print(f"  positive fragments: {pos}/{len(pnls)} ({100*pos/len(pnls):.0f}%)")
        print(f"  negative fragments: {neg}/{len(pnls)} ({100*neg/len(pnls):.0f}%)")
        loss_sum = sum(p for p in pnls if p < 0)
        print(f"  loss-fragment sum: {loss_sum:>+10,.0f}  (worst-case sub PnL)")


if __name__ == "__main__":
    main()
