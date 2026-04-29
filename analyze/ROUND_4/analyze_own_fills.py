"""Run a backtest and aggregate our own fills by counterparty bot.

Shows for each (bot, product) pair: how many fills, total qty, signed net,
and average price-vs-mid edge of those fills. Reveals which bots we are
actually intercepting, and which are immune to our quote placement.

Usage:
    python analyze_own_fills.py [algo_path] [round day]
Default:
    python analyze_own_fills.py algorithms/round_4_botflow.py 4 1
"""
from __future__ import annotations
import io
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import backtester as BT


def _mid_lookup(price_data):
    out = {}
    for ts, products in price_data.items():
        for sym, pr in products.items():
            out[(ts, sym)] = pr.mid_price
    return out


def run(algo_path: str, round_num: int, day_num: int):
    mod = BT.load_algorithm(Path(algo_path))
    reader = BT.DataReader(REPO_ROOT / "data")
    price_data = reader.read_prices(round_num, day_num)
    mid_lkup = _mid_lookup(price_data)

    with contextlib.redirect_stdout(io.StringIO()):
        result = BT.run_backtest(mod, reader, round_num, day_num,
                                 print_output=False)
    log = result["log"]
    own_trades = [t for t in log["tradeHistory"]
                  if t.get("buyer") == "SUBMISSION" or t.get("seller") == "SUBMISSION"]

    # Aggregate
    by_bot_prod = defaultdict(lambda: {
        "buy_n": 0, "buy_qty": 0, "sell_n": 0, "sell_qty": 0,
        "edge_sum": 0.0, "edge_n": 0,
    })
    counterparty_unknown = defaultdict(lambda: {
        "buy_qty": 0, "sell_qty": 0,
    })
    total_volume = 0
    for t in own_trades:
        sym = t["symbol"]
        qty = int(t["quantity"])
        price = float(t["price"])
        total_volume += qty
        we_bought = (t["buyer"] == "SUBMISSION")
        cp = t["seller"] if we_bought else t["buyer"]
        m = mid_lkup.get((t["timestamp"], sym), None)
        if not cp:
            tgt = counterparty_unknown[sym]
            if we_bought:
                tgt["buy_qty"] += qty
            else:
                tgt["sell_qty"] += qty
            continue
        rec = by_bot_prod[(cp, sym)]
        if we_bought:
            rec["buy_n"] += 1; rec["buy_qty"] += qty
        else:
            rec["sell_n"] += 1; rec["sell_qty"] += qty
        if m is not None:
            edge = (m - price) if we_bought else (price - m)
            rec["edge_sum"] += edge * qty
            rec["edge_n"] += qty

    print(f"Algo={algo_path}  round={round_num}  day={day_num}")
    print(f"PnL: {result['profit']:.0f}   "
          f"own_fills: {len(own_trades)}   total_qty: {total_volume}")
    print()
    print("PER-COUNTERPARTY FILL ATTRIBUTION (Phase 2 only — Phase 1 is anonymous)")
    print("=" * 100)
    print(f"{'counterparty':<12} {'product':<24} "
          f"{'fills':>6} {'qty':>6} {'buyQ':>6} {'sellQ':>6} {'net':>6} {'avgEdge':>9}")
    print("-" * 100)
    rows = []
    for (cp, sym), r in by_bot_prod.items():
        total_q = r["buy_qty"] + r["sell_qty"]
        net = r["buy_qty"] - r["sell_qty"]
        edge = r["edge_sum"] / r["edge_n"] if r["edge_n"] else 0.0
        rows.append((cp, sym, r["buy_n"] + r["sell_n"], total_q,
                     r["buy_qty"], r["sell_qty"], net, edge))
    rows.sort(key=lambda r: -r[3])
    for cp, sym, n, q, bq, sq, net, edge in rows:
        print(f"{cp:<12} {sym:<24} {n:>6} {q:>6} {bq:>6} {sq:>6} {net:>+6} {edge:>+9.2f}")

    if counterparty_unknown:
        print()
        print("ANONYMOUS FILLS (Phase 1 — order book has no bot identity)")
        print("-" * 60)
        print(f"{'product':<24} {'buyQ':>8} {'sellQ':>8}")
        for sym, r in sorted(counterparty_unknown.items()):
            print(f"{sym:<24} {r['buy_qty']:>8} {r['sell_qty']:>8}")

    # Per-bot summary across products
    print()
    print("PER-BOT SUMMARY")
    print("-" * 80)
    by_bot = defaultdict(lambda: {"q": 0, "buy_q": 0, "sell_q": 0, "n": 0,
                                  "edge_sum": 0.0, "edge_n": 0})
    for (cp, sym), r in by_bot_prod.items():
        b = by_bot[cp]
        b["q"] += r["buy_qty"] + r["sell_qty"]
        b["buy_q"] += r["buy_qty"]
        b["sell_q"] += r["sell_qty"]
        b["n"] += r["buy_n"] + r["sell_n"]
        b["edge_sum"] += r["edge_sum"]
        b["edge_n"] += r["edge_n"]
    print(f"{'bot':<12} {'fills':>6} {'qty':>6} {'buyQ':>6} {'sellQ':>6} "
          f"{'net':>6} {'avgEdge':>9}")
    bot_rows = []
    for cp, b in by_bot.items():
        edge = b["edge_sum"] / b["edge_n"] if b["edge_n"] else 0.0
        bot_rows.append((cp, b["n"], b["q"], b["buy_q"], b["sell_q"],
                         b["buy_q"] - b["sell_q"], edge))
    bot_rows.sort(key=lambda r: -r[2])
    for cp, n, q, bq, sq, net, edge in bot_rows:
        print(f"{cp:<12} {n:>6} {q:>6} {bq:>6} {sq:>6} {net:>+6} {edge:>+9.2f}")


if __name__ == "__main__":
    algo = sys.argv[1] if len(sys.argv) > 1 else "algorithms/round_4_botflow.py"
    rnd = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    day = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    run(algo, rnd, day)
