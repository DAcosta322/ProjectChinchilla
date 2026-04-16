"""
Compare DUMMY (no orders) vs BEST ALGO submission logs for ASH_COATED_OSMIUM.
Answers: does our presence inside the spread create more bot trading?
"""

import json
from collections import Counter, defaultdict

DUMMY_PATH = "d:/UW/1B/prosperity_imc/dump/ROUND_1_DUMMY/228836.log"
ACTIVE_PATH = "d:/UW/1B/prosperity_imc/dump/172012/172012.log"
SYMBOL = "ASH_COATED_OSMIUM"


def load_log(path):
    with open(path, "r") as f:
        return json.load(f)


def parse_activities(data, symbol):
    """Parse activitiesLog CSV for a given symbol. Returns list of dicts."""
    lines = data["activitiesLog"].strip().split("\n")
    header = lines[0].split(";")
    rows = []
    for line in lines[1:]:
        parts = line.split(";")
        if parts[2] != symbol:
            continue
        row = {}
        for i, h in enumerate(header):
            val = parts[i] if i < len(parts) else ""
            if h in ("day", "timestamp", "bid_volume_1", "bid_volume_2", "bid_volume_3",
                      "ask_volume_1", "ask_volume_2", "ask_volume_3"):
                row[h] = int(val) if val else None
            elif h in ("bid_price_1", "bid_price_2", "bid_price_3",
                        "ask_price_1", "ask_price_2", "ask_price_3",
                        "mid_price", "profit_and_loss"):
                row[h] = float(val) if val else None
            else:
                row[h] = val
        rows.append(row)
    return rows


def parse_trades(data, symbol):
    """Extract trades for a symbol from tradeHistory."""
    return [t for t in data["tradeHistory"] if t["symbol"] == symbol]


def classify_trade(trade):
    """Classify a trade as: bot-bot, our-buy, our-sell."""
    buyer = trade["buyer"]
    seller = trade["seller"]
    if buyer == "SUBMISSION":
        return "our_buy"
    elif seller == "SUBMISSION":
        return "our_sell"
    else:
        return "bot_bot"


def spread_stats(activities):
    """Compute spread statistics from activity rows."""
    spreads = []
    for r in activities:
        b1 = r.get("bid_price_1")
        a1 = r.get("ask_price_1")
        if b1 is not None and a1 is not None:
            spreads.append(a1 - b1)
    if not spreads:
        return {}
    spreads.sort()
    n = len(spreads)
    return {
        "mean": sum(spreads) / n,
        "median": spreads[n // 2],
        "min": min(spreads),
        "max": max(spreads),
        "p25": spreads[n // 4],
        "p75": spreads[3 * n // 4],
        "count": n,
    }


def price_distribution(trades):
    """Return Counter of prices."""
    return Counter(t["price"] for t in trades)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
dummy_data = load_log(DUMMY_PATH)
active_data = load_log(ACTIVE_PATH)

dummy_activities = parse_activities(dummy_data, SYMBOL)
active_activities = parse_activities(active_data, SYMBOL)

dummy_trades = parse_trades(dummy_data, SYMBOL)
active_trades = parse_trades(active_data, SYMBOL)

# Days present
dummy_days = sorted(set(r["day"] for r in dummy_activities))
active_days = sorted(set(r["day"] for r in active_activities))

# ---------------------------------------------------------------------------
# Classify trades
# ---------------------------------------------------------------------------
dummy_classified = defaultdict(list)
for t in dummy_trades:
    dummy_classified[classify_trade(t)].append(t)

active_classified = defaultdict(list)
for t in active_trades:
    active_classified[classify_trade(t)].append(t)

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
SEP = "=" * 80

print(SEP)
print(f"  COMPARISON: DUMMY vs BEST ALGO  --  {SYMBOL}")
print(SEP)
print()

print(f"Days in DUMMY log:  {dummy_days}")
print(f"Days in ACTIVE log: {active_days}")
print()

# --- Trade counts ---
print(SEP)
print("  1. TRADE COUNT COMPARISON")
print(SEP)
print(f"{'Metric':<40} {'DUMMY':>10} {'ACTIVE':>10} {'Delta':>10}")
print("-" * 70)

dummy_bot = dummy_classified["bot_bot"]
active_bot = active_classified["bot_bot"]
active_our_buy = active_classified["our_buy"]
active_our_sell = active_classified["our_sell"]

rows = [
    ("Total trades", len(dummy_trades), len(active_trades)),
    ("Bot-bot trades", len(dummy_bot), len(active_bot)),
    ("Our buys (we are buyer)", 0, len(active_our_buy)),
    ("Our sells (we are seller)", 0, len(active_our_sell)),
    ("Total volume (qty)", sum(t["quantity"] for t in dummy_trades),
     sum(t["quantity"] for t in active_trades)),
    ("Bot-bot volume", sum(t["quantity"] for t in dummy_bot),
     sum(t["quantity"] for t in active_bot)),
    ("Our buy volume", 0, sum(t["quantity"] for t in active_our_buy)),
    ("Our sell volume", 0, sum(t["quantity"] for t in active_our_sell)),
]
for label, d, a in rows:
    delta = a - d
    sign = "+" if delta > 0 else ""
    print(f"{label:<40} {d:>10} {a:>10} {sign + str(delta):>10}")

print()
print("KEY FINDING: Bot-bot trades in DUMMY vs ACTIVE tells us if our")
print("presence CREATES additional bot trading or merely intercepts it.")
print()

# --- Timestamp matching ---
print(SEP)
print("  2. MATCHED-TIMESTAMP ANALYSIS")
print(SEP)
print()

dummy_ts = {t["timestamp"]: t for t in dummy_bot}
active_bot_ts = {t["timestamp"]: t for t in active_bot}

# Bot trades at same timestamps
common_ts = set(dummy_ts.keys()) & set(active_bot_ts.keys())
dummy_only_ts = set(dummy_ts.keys()) - set(active_bot_ts.keys())
active_only_bot_ts = set(active_bot_ts.keys()) - set(dummy_ts.keys())

print(f"Bot-bot trades at same timestamp in both:  {len(common_ts)}")
print(f"Bot-bot trades ONLY in DUMMY:              {len(dummy_only_ts)}")
print(f"Bot-bot trades ONLY in ACTIVE:             {len(active_only_bot_ts)}")
print()

# For timestamps that exist in dummy, check if active has a SUBMISSION trade instead
active_all_by_ts = defaultdict(list)
for t in active_trades:
    active_all_by_ts[t["timestamp"]].append(t)

intercepted = 0
intercepted_details = []
for ts in sorted(dummy_only_ts):
    if ts in active_all_by_ts:
        for at in active_all_by_ts[ts]:
            if at["buyer"] == "SUBMISSION" or at["seller"] == "SUBMISSION":
                intercepted += 1
                dt = dummy_ts[ts]
                intercepted_details.append((ts, dt["price"], at["price"],
                                            "buy" if at["buyer"] == "SUBMISSION" else "sell"))
                break

print(f"Dummy bot-bot trades INTERCEPTED by us in active: {intercepted}/{len(dummy_only_ts)}")
if intercepted_details:
    print(f"  {'Timestamp':>10}  {'Dummy Price':>12}  {'Our Price':>12}  {'Our Side':>10}")
    for ts, dp, ap, side in intercepted_details[:20]:
        print(f"  {ts:>10}  {dp:>12.0f}  {ap:>12.0f}  {side:>10}")
    if len(intercepted_details) > 20:
        print(f"  ... and {len(intercepted_details) - 20} more")
print()

# New trades that only exist in active (with SUBMISSION)
active_submission_only_ts = set()
for t in active_trades:
    if (t["buyer"] == "SUBMISSION" or t["seller"] == "SUBMISSION"):
        if t["timestamp"] not in dummy_ts:
            active_submission_only_ts.add(t["timestamp"])

new_our_trades = [t for t in active_trades
                  if (t["buyer"] == "SUBMISSION" or t["seller"] == "SUBMISSION")
                  and t["timestamp"] not in dummy_ts]
print(f"Our trades at timestamps with NO dummy bot trade: {len(new_our_trades)}")
print("  (These are trades we CREATED by providing liquidity inside the spread)")
if new_our_trades:
    print(f"  {'Timestamp':>10}  {'Price':>8}  {'Qty':>5}  {'Side':>6}")
    for t in new_our_trades[:15]:
        side = "buy" if t["buyer"] == "SUBMISSION" else "sell"
        print(f"  {t['timestamp']:>10}  {t['price']:>8.0f}  {t['quantity']:>5}  {side:>6}")
    if len(new_our_trades) > 15:
        print(f"  ... and {len(new_our_trades) - 15} more")
print()

# --- Price analysis ---
print(SEP)
print("  3. PRICE ANALYSIS")
print(SEP)
print()

print("Bot-bot trade prices in DUMMY:")
dummy_bot_prices = price_distribution(dummy_bot)
for price in sorted(dummy_bot_prices.keys()):
    print(f"  {price:>10.0f}  x{dummy_bot_prices[price]}")

print()
print("Bot-bot trade prices in ACTIVE:")
active_bot_prices = price_distribution(active_bot)
if active_bot:
    for price in sorted(active_bot_prices.keys()):
        print(f"  {price:>10.0f}  x{active_bot_prices[price]}")
else:
    print("  (none -- all bot trades were intercepted by us)")

print()
print("Our trade prices in ACTIVE:")
our_trades = [t for t in active_trades if t["buyer"] == "SUBMISSION" or t["seller"] == "SUBMISSION"]
our_buy_prices = Counter(t["price"] for t in our_trades if t["buyer"] == "SUBMISSION")
our_sell_prices = Counter(t["price"] for t in our_trades if t["seller"] == "SUBMISSION")

print("  BUYS (we bought from bots):")
for price in sorted(our_buy_prices.keys()):
    print(f"    {price:>10.0f}  x{our_buy_prices[price]}")

print("  SELLS (we sold to bots):")
for price in sorted(our_sell_prices.keys()):
    print(f"    {price:>10.0f}  x{our_sell_prices[price]}")

print()

# Average buy/sell prices
if active_our_buy:
    avg_buy = sum(t["price"] * t["quantity"] for t in active_our_buy) / sum(t["quantity"] for t in active_our_buy)
    print(f"  VWAP of our buys:  {avg_buy:.2f}")
if active_our_sell:
    avg_sell = sum(t["price"] * t["quantity"] for t in active_our_sell) / sum(t["quantity"] for t in active_our_sell)
    print(f"  VWAP of our sells: {avg_sell:.2f}")
if active_our_buy and active_our_sell:
    print(f"  VWAP spread earned: {avg_sell - avg_buy:.2f}")
print()

# --- Spread analysis ---
print(SEP)
print("  4. SPREAD DISTRIBUTION COMPARISON")
print(SEP)
print()

dummy_spread = spread_stats(dummy_activities)
active_spread = spread_stats(active_activities)

print(f"{'Metric':<20} {'DUMMY':>10} {'ACTIVE':>10}")
print("-" * 40)
for key in ["mean", "median", "min", "max", "p25", "p75"]:
    dv = dummy_spread.get(key, 0)
    av = active_spread.get(key, 0)
    print(f"{key:<20} {dv:>10.1f} {av:>10.1f}")

print()

# Spread distribution buckets
def spread_histogram(activities):
    counts = Counter()
    for r in activities:
        b1 = r.get("bid_price_1")
        a1 = r.get("ask_price_1")
        if b1 is not None and a1 is not None:
            counts[a1 - b1] += 1
    return counts

dummy_sh = spread_histogram(dummy_activities)
active_sh = spread_histogram(active_activities)
all_spreads_vals = sorted(set(list(dummy_sh.keys()) + list(active_sh.keys())))

print("Spread value distribution (count of ticks at each spread):")
print(f"  {'Spread':>8}  {'DUMMY':>8}  {'ACTIVE':>8}  {'Delta':>8}")
for s in all_spreads_vals:
    d = dummy_sh.get(s, 0)
    a = active_sh.get(s, 0)
    delta = a - d
    sign = "+" if delta > 0 else ""
    print(f"  {s:>8.0f}  {d:>8}  {a:>8}  {sign + str(delta):>8}")
print()

# --- PnL from osmium ---
print(SEP)
print("  5. OUR PNL FROM OSMIUM (ACTIVE ALGO)")
print(SEP)
print()

# Final PnL from activitiesLog
final_pnl = active_activities[-1].get("profit_and_loss", 0)
print(f"Final PnL from activitiesLog: {final_pnl:.2f}")

# Reconstruct PnL from trades
position = 0
cash = 0.0
for t in sorted(our_trades, key=lambda x: x["timestamp"]):
    if t["buyer"] == "SUBMISSION":
        position += t["quantity"]
        cash -= t["price"] * t["quantity"]
    else:
        position -= t["quantity"]
        cash += t["price"] * t["quantity"]

mid_price = active_activities[-1].get("mid_price", 10000)
mark_to_market = cash + position * mid_price
print(f"Reconstructed from trades:")
print(f"  Cash flow:     {cash:>12.2f}")
print(f"  Final position: {position:>5}")
print(f"  Final mid:      {mid_price:.1f}")
print(f"  Mark-to-market: {mark_to_market:.2f}")
print()

# --- Fill classification ---
print(SEP)
print("  6. FILL CLASSIFICATION (ACTIVE ALGO)")
print(SEP)
print()

# Build activity lookup: timestamp -> (bid1, ask1)
act_lookup = {}
for r in active_activities:
    act_lookup[r["timestamp"]] = (r.get("bid_price_1"), r.get("ask_price_1"))

# For each of our trades, check if we took existing liquidity or got filled passively
# If our buy price == ask1 at that timestamp, we took the ask (aggressive)
# If our buy price < ask1, we got filled passively (bot hit our bid)
# Similarly for sells
aggressive = []
passive = []
ambiguous = []

for t in our_trades:
    ts = t["timestamp"]
    # Look at the PREVIOUS tick's book (order book before our orders were placed)
    prev_ts = ts - 100  # ticks are every 100
    book = act_lookup.get(prev_ts) or act_lookup.get(ts)
    if book is None:
        ambiguous.append(t)
        continue
    bid1, ask1 = book
    if t["buyer"] == "SUBMISSION":
        # We bought
        if bid1 and t["price"] <= bid1:
            # We bought at or below the existing bid -- impossible if passive, so this is aggressive
            # Actually if we buy at bid, that could mean we placed a bid and bot hit it
            # Let me reconsider: if our buy price >= ask1, we took the ask (aggressive)
            pass
        if ask1 and t["price"] >= ask1:
            aggressive.append(("buy", t["price"], ask1, t["quantity"], ts))
        elif bid1 and t["price"] <= bid1 + 1:
            # Bought near bid -- likely passive fill (bot sold to our bid)
            passive.append(("buy", t["price"], bid1, t["quantity"], ts))
        else:
            # Price is between bid and ask -- we placed inside spread, bot hit us
            passive.append(("buy", t["price"], bid1, t["quantity"], ts))
    else:
        # We sold
        if bid1 and t["price"] <= bid1:
            aggressive.append(("sell", t["price"], bid1, t["quantity"], ts))
        elif ask1 and t["price"] >= ask1 - 1:
            passive.append(("sell", t["price"], ask1, t["quantity"], ts))
        else:
            passive.append(("sell", t["price"], ask1, t["quantity"], ts))

print(f"Aggressive fills (we took existing book): {len(aggressive)}")
print(f"Passive fills (bot traded against our order): {len(passive)}")
print(f"Ambiguous: {len(ambiguous)}")
print()

agg_vol = sum(x[3] for x in aggressive)
pas_vol = sum(x[3] for x in passive)
amb_vol = sum(t["quantity"] for t in ambiguous)
print(f"Aggressive volume: {agg_vol}")
print(f"Passive volume:    {pas_vol}")
print(f"Ambiguous volume:  {amb_vol}")
print()

if aggressive:
    print("Aggressive fills detail (first 10):")
    print(f"  {'Side':>6}  {'Price':>8}  {'Book':>8}  {'Qty':>5}  {'TS':>8}")
    for side, price, ref, qty, ts in aggressive[:10]:
        print(f"  {side:>6}  {price:>8.0f}  {ref:>8.0f}  {qty:>5}  {ts:>8}")

if passive:
    print("Passive fills detail (first 10):")
    print(f"  {'Side':>6}  {'Price':>8}  {'Book':>8}  {'Qty':>5}  {'TS':>8}")
    for side, price, ref, qty, ts in passive[:10]:
        print(f"  {side:>6}  {price:>8.0f}  {ref:>8.0f}  {qty:>5}  {ts:>8}")

print()

# --- Summary ---
print(SEP)
print("  SUMMARY")
print(SEP)
print()
print(f"Bot-bot trades:  DUMMY={len(dummy_bot)}, ACTIVE={len(active_bot)}, "
      f"delta={len(active_bot) - len(dummy_bot)}")
print(f"  -> In ACTIVE, {len(dummy_bot) - len(active_bot)} bot-bot trades were intercepted by us")
print(f"  -> {len(new_our_trades)} of our trades are at NEW timestamps (liquidity we created)")
print()

bot_vol_dummy = sum(t["quantity"] for t in dummy_bot)
bot_vol_active = sum(t["quantity"] for t in active_bot)
our_vol = sum(t["quantity"] for t in our_trades)
print(f"Total bot-bot volume:  DUMMY={bot_vol_dummy}, ACTIVE={bot_vol_active}")
print(f"Our total volume:      {our_vol}")
print(f"Total market volume:   DUMMY={bot_vol_dummy}, ACTIVE={bot_vol_active + our_vol}")
print()
print(f"Osmium PnL: {final_pnl:.2f}")
print(f"Avg spread in DUMMY:  {dummy_spread['mean']:.1f}")
print(f"Avg spread in ACTIVE: {active_spread['mean']:.1f}")
print()

if len(active_bot) < len(dummy_bot):
    print("CONCLUSION: Our presence REPLACES bot-bot trades (we intercept them).")
    print("We do NOT create additional bot-bot trading -- we capture the existing flow.")
elif len(active_bot) > len(dummy_bot):
    print("CONCLUSION: Our presence INCREASES bot-bot trading!")
    print("Providing liquidity inside the spread stimulates additional bot activity.")
else:
    print("CONCLUSION: Our presence does NOT change the number of bot-bot trades.")
    print("We trade alongside existing bot flow without displacing or creating it.")

if new_our_trades:
    print()
    print(f"HOWEVER: We generated {len(new_our_trades)} trades at timestamps where NO bot-bot")
    print("trade existed in the dummy. This means bots DO react to our quotes inside the spread.")
