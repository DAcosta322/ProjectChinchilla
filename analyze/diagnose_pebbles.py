"""Diagnose per-product trading behavior for the pebbles BT."""
import json
import sys
from pathlib import Path
from collections import defaultdict


def diag(log_path: Path):
    with open(log_path) as f:
        log = json.load(f)
    trades = log["tradeHistory"]
    by_p = defaultdict(lambda: {"b": 0, "bq": 0, "bv": 0.0, "s": 0, "sq": 0, "sv": 0.0})
    for t in trades:
        p = t["symbol"]
        s = by_p[p]
        if t["buyer"] == "SUBMISSION":
            s["b"] += 1
            s["bq"] += t["quantity"]
            s["bv"] += t["price"] * t["quantity"]
        else:
            s["s"] += 1
            s["sq"] += t["quantity"]
            s["sv"] += t["price"] * t["quantity"]
    print(f"{log_path.name}")
    print(f"  total trades: {len(trades)}")
    print(f"  {'product':14}{'buys':>6}{'qty':>5}{'avgPx':>9}{'sells':>6}{'qty':>5}{'avgPx':>9}{'spread/RT':>11}")
    for p in sorted(by_p):
        st = by_p[p]
        avgB = st["bv"] / st["bq"] if st["bq"] else 0
        avgS = st["sv"] / st["sq"] if st["sq"] else 0
        spread = avgS - avgB if (st["bq"] and st["sq"]) else 0
        print(f"  {p:14}{st['b']:>6}{st['bq']:>5}{avgB:>9.2f}{st['s']:>6}{st['sq']:>5}{avgS:>9.2f}{spread:>11.2f}")


def main():
    if len(sys.argv) > 1:
        for p in sys.argv[1:]:
            diag(Path(p))
            print()
    else:
        # Find latest pebbles run
        logs = sorted(Path("logs/ROUND_5").glob("round_5_pebbles*"), reverse=True)
        if not logs:
            print("no logs found")
            return
        run_dir = logs[0]
        for log_file in sorted(run_dir.glob("*.log")):
            diag(log_file)
            print()


if __name__ == "__main__":
    main()
