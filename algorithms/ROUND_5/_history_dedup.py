"""History-keeping helper for IMC platform algos.

The IMC platform keeps state.own_trades alive for multiple ticks, contrary
to documentation that claims "trades since last TradingState came in".
Algos that process own_trades must dedupe by Trade.timestamp or risk
state-refresh loops (sub 554229 made $14,925 vs $27,190 with the dedup fix).

Usage from a Trader.run(state) call:

    from _history_dedup import process_own_trades

    last_processed_ts = data.get("last_processed_ts", -1)
    new_buys, new_sells, last_processed_ts = process_own_trades(
        state, last_processed_ts, products_set,
    )
    for p, t in new_buys:
        # handle a fresh BUY at price t.price, ts t.timestamp
        ...
    for p, t in new_sells:
        ...

    # persist
    data["last_processed_ts"] = last_processed_ts

NOTE: this file would be a SHARED module if IMC supported imports across
files in submitted code. Since IMC submissions are single-file, treat this
as a CANONICAL REFERENCE — copy the dedup pattern into each algo file.
"""

from typing import Iterable, Tuple, List


def process_own_trades(state, last_processed_ts: int, products: Iterable[str]):
    """Iterate own_trades, returning fresh trades (deduped by timestamp).

    Returns:
        (new_buys, new_sells, new_last_processed_ts) where:
          - new_buys: list of (product, Trade) where buyer == 'SUBMISSION'
          - new_sells: list of (product, Trade) where seller == 'SUBMISSION'
          - new_last_processed_ts: max timestamp seen across new trades
            (fall back to last_processed_ts if no new trades)
    """
    products_set = set(products)
    max_seen_ts = last_processed_ts
    new_buys: List = []
    new_sells: List = []
    if not (hasattr(state, "own_trades") and state.own_trades):
        return new_buys, new_sells, max_seen_ts
    for p, trades in state.own_trades.items():
        if p not in products_set or not trades:
            continue
        for t in trades:
            t_ts = getattr(t, "timestamp", 0)
            if t_ts <= last_processed_ts:
                continue  # already processed in a prior call
            if t_ts > max_seen_ts:
                max_seen_ts = t_ts
            if getattr(t, "buyer", None) == "SUBMISSION":
                new_buys.append((p, t))
            elif getattr(t, "seller", None) == "SUBMISSION":
                new_sells.append((p, t))
    return new_buys, new_sells, max_seen_ts


# === CANONICAL DEDUP PATTERN (paste into algo file) ===
DEDUP_PATTERN = """
# In Trader.run() — at top, after loading traderData:
last_processed_ts = data.get("last_processed_ts", -1)

# Replace any `for t in state.own_trades[p]: ...` with this guarded loop:
max_seen_ts = last_processed_ts
if hasattr(state, "own_trades") and state.own_trades:
    for p, trades in state.own_trades.items():
        if p not in YOUR_PRODUCTS_SET or not trades:
            continue
        for t in trades:
            t_ts = getattr(t, "timestamp", 0)
            if t_ts <= last_processed_ts:
                continue
            if t_ts > max_seen_ts:
                max_seen_ts = t_ts
            # Use t.buyer/t.seller fields, NOT qty sign (always positive in IMC).
            if getattr(t, "buyer", None) == "SUBMISSION":
                # ... handle BUY ...
                pass
            elif getattr(t, "seller", None) == "SUBMISSION":
                # ... handle SELL ...
                pass
last_processed_ts = max_seen_ts

# At end of run(), persist:
data["last_processed_ts"] = last_processed_ts
"""
