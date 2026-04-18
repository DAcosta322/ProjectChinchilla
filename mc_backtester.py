"""Monte Carlo backtester.

Fits stochastic models to historical data and generates N synthetic
realizations. Evaluates strategy across the distribution, not one path.

Why: the deterministic backtester replays historical prices verbatim, so
execution-quality strategies whose value depends on mean-reversion playing
out on average (e.g., URGENCY) can't be scored correctly on a single path.
Averaging over paths lets the distributional edge show up.

Models fit per product:
- OU mean-reversion for bounded products (osmium)
- Linear drift + noise for trending products (pepper)
- Book shape sampled from historical spread/volume distributions
- Bot flow sampled from historical arrival rate + side bias by recent drift

Usage:
    python mc_backtester.py <algo_path> [--round R] [--day D] [--paths N] [--seed S]
"""

from __future__ import annotations
import argparse
import random
import statistics
import sys
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from datamodel import Listing, OrderDepth, Trade, TradingState, Order, Observation
from backtester import (
    DataReader, MarketTrade, OrderMatcher, enforce_position_limits,
    POSITION_LIMITS, CURRENCY,
)


# ---------------------------------------------------------------------------
# Market model: fit + generate
# ---------------------------------------------------------------------------
class MarketModel:
    """Fitted stochastic model for one product.

    Mid dynamics:
        OU:    mid[t+1] = mid[t] + kappa*(anchor - mid[t]) + drift + N(0, sigma)
        Auto-selects OU when historical range is bounded (< 100) around mean.
    Book synthesis:
        Spread sampled from historical spread distribution.
        Level volumes sampled from historical per-tick level patterns.
    Flow synthesis:
        Per-tick Bernoulli arrival at historical rate.
        Side weighted by recent drift (more buys when rising).
    """

    def __init__(self, product: str):
        self.product = product
        self.mid_start = 10000
        self.n_ticks = 0
        self.drift = 0.0            # mean increment per tick
        self.kappa = 0.0            # OU mean-reversion speed (0 = pure drift)
        self.anchor = None
        self.sigma = 1.0            # noise stdev per tick
        self.spread_hist: List[int] = []
        self.bid_patterns: List[List[int]] = []   # list of bid volumes per level
        self.ask_patterns: List[List[int]] = []
        self.flow_rate = 0.0
        self.flow_size_hist: List[int] = []
        self.flow_side_ratio = 0.5                # fraction of flow that is buy-side

    def fit(self, price_rows, market_trades, timestamps):
        mids = [pr.mid_price for pr in price_rows if pr.mid_price > 0]
        if not mids:
            return
        self.mid_start = mids[0]
        self.n_ticks = len(price_rows)

        diffs = [mids[i] - mids[i - 1] for i in range(1, len(mids))]
        if not diffs:
            return
        mean_diff = sum(diffs) / len(diffs)
        mid_range = max(mids) - min(mids)

        # Detect OU: range bounded relative to mid magnitude (osmium ~10000 ±15)
        if mid_range < 60:
            self.anchor = sum(mids) / len(mids)
            # Fit kappa: Δmid_t ≈ kappa * (anchor - mid_t)
            deviations = [self.anchor - mids[i - 1] for i in range(1, len(mids))]
            num = sum(d * x for d, x in zip(diffs, deviations))
            den = sum(x * x for x in deviations) or 1.0
            self.kappa = max(0.0, min(1.0, num / den))
            residuals = [diffs[i] - self.kappa * deviations[i]
                         for i in range(len(diffs))]
            self.drift = sum(residuals) / len(residuals)
            self.sigma = statistics.pstdev(residuals) or 1.0
        else:
            self.kappa = 0.0
            self.anchor = None
            self.drift = mean_diff
            self.sigma = statistics.pstdev(diffs) or 1.0

        # Book shape patterns
        for pr in price_rows:
            if pr.bid_prices and pr.ask_prices:
                sp = pr.ask_prices[0] - pr.bid_prices[0]
                if sp > 0:
                    self.spread_hist.append(sp)
            if pr.bid_volumes:
                self.bid_patterns.append(list(pr.bid_volumes))
            if pr.ask_volumes:
                self.ask_patterns.append(list(pr.ask_volumes))
        if not self.spread_hist:
            self.spread_hist = [14]
        if not self.bid_patterns:
            self.bid_patterns = [[20, 15]]
        if not self.ask_patterns:
            self.ask_patterns = [[20, 15]]

        # Flow rate and size
        if market_trades:
            self.flow_rate = len(market_trades) / max(1, self.n_ticks)
            self.flow_size_hist = [abs(t.quantity) for t in market_trades]
            # Side bias from historical trades vs mid at that tick
            ts_to_mid = {pr.timestamp: pr.mid_price for pr in price_rows if pr.mid_price > 0}
            buy_count = sell_count = 0
            for t in market_trades:
                m = ts_to_mid.get(t.timestamp)
                if m is None:
                    continue
                if t.price > m:
                    buy_count += 1
                elif t.price < m:
                    sell_count += 1
            total = buy_count + sell_count
            if total > 0:
                self.flow_side_ratio = buy_count / total

    def generate_mid_path(self, rng: random.Random) -> List[float]:
        mid = float(self.mid_start)
        path = [mid]
        for _ in range(self.n_ticks - 1):
            if self.kappa > 0 and self.anchor is not None:
                mid = mid + self.kappa * (self.anchor - mid) + self.drift + rng.gauss(0, self.sigma)
            else:
                mid = mid + self.drift + rng.gauss(0, self.sigma)
            path.append(mid)
        return path

    def generate_book(self, mid: float, rng: random.Random) -> OrderDepth:
        od = OrderDepth()
        spread = rng.choice(self.spread_hist)
        bid1 = int(round(mid - spread / 2))
        ask1 = bid1 + spread
        bid_vols = rng.choice(self.bid_patterns)
        ask_vols = rng.choice(self.ask_patterns)
        for i, vol in enumerate(bid_vols[:3]):
            od.buy_orders[bid1 - i] = int(vol)
        for i, vol in enumerate(ask_vols[:3]):
            od.sell_orders[ask1 + i] = -int(vol)
        return od

    def generate_flow(self, mid: float, recent_drift: float,
                      ts: int, rng: random.Random) -> List[MarketTrade]:
        trades = []
        if rng.random() >= self.flow_rate:
            return trades
        qty = rng.choice(self.flow_size_hist) if self.flow_size_hist else 5
        # Side bias: baseline ratio shifted by recent drift sign
        buy_prob = self.flow_side_ratio
        if recent_drift > 0:
            buy_prob = min(1.0, buy_prob + 0.15)
        elif recent_drift < 0:
            buy_prob = max(0.0, buy_prob - 0.15)
        if rng.random() < buy_prob:
            # Aggressive buy: price at/above mid
            price = mid + rng.uniform(0, 8)
        else:
            price = mid - rng.uniform(0, 8)
        trades.append(MarketTrade(
            timestamp=ts, buyer="bot", seller="bot",
            symbol=self.product, price=float(round(price)), quantity=int(qty),
        ))
        return trades


# ---------------------------------------------------------------------------
# Simulation runner
# ---------------------------------------------------------------------------
def _build_state(timestamp, products, position, own_trades,
                 market_trades_at_ts, order_depths, trader_data):
    listings = {p: Listing(symbol=p, product=p, denomination=CURRENCY)
                for p in products}
    mt_dict: Dict[str, List[Trade]] = {}
    for product, mt_list in market_trades_at_ts.items():
        mt_dict[product] = [
            Trade(symbol=mt.symbol, price=int(mt.price),
                  quantity=mt.quantity, buyer=mt.buyer,
                  seller=mt.seller, timestamp=mt.timestamp)
            for mt in mt_list
        ]
    return TradingState(
        traderData=trader_data, timestamp=timestamp,
        listings=listings, order_depths=order_depths,
        own_trades=own_trades, market_trades=mt_dict,
        position=dict(position),
        observations=Observation({}, {}),
    )


def run_one_path(trader_module, models, timestamps, products,
                 rng: random.Random) -> Dict[str, float]:
    position = {p: 0 for p in products}
    pnl = {p: 0.0 for p in products}
    own_trades = {p: [] for p in products}
    trader_data = ""
    matcher = OrderMatcher(position, pnl)
    trader = trader_module.Trader()

    mid_paths = {p: models[p].generate_mid_path(rng) for p in products}

    for i, ts in enumerate(timestamps):
        order_depths = {}
        market_trades_at_ts = {}
        for product in products:
            mid = mid_paths[product][i] if i < len(mid_paths[product]) else models[product].mid_start
            recent_drift = 0.0
            if i >= 5:
                recent_drift = mid - mid_paths[product][i - 5]
            order_depths[product] = models[product].generate_book(mid, rng)
            market_trades_at_ts[product] = models[product].generate_flow(
                mid, recent_drift, ts, rng)

        state = _build_state(ts, products, position, own_trades,
                             market_trades_at_ts, order_depths, trader_data)

        try:
            result = trader.run(state)
        except Exception:
            result = ({}, 0, trader_data)
        if isinstance(result, tuple):
            if len(result) == 3:
                orders_dict, _, trader_data = result
            elif len(result) == 2:
                orders_dict, trader_data = result
            else:
                orders_dict = result[0] if result else {}
        else:
            orders_dict = result if isinstance(result, dict) else {}
        if orders_dict is None:
            orders_dict = {}

        valid = enforce_position_limits(orders_dict, position, POSITION_LIMITS)
        new_trades = matcher.match(valid, order_depths, ts, market_trades_at_ts)

        own_trades = {p: [] for p in products}
        for t in new_trades:
            tr = Trade(symbol=t["symbol"], price=int(t["price"]),
                       quantity=t["quantity"], buyer=t["buyer"],
                       seller=t["seller"], timestamp=ts)
            own_trades.setdefault(t["symbol"], []).append(tr)

    # Mark-to-market final positions at final mid
    for product in products:
        final_mid = mid_paths[product][-1] if mid_paths[product] else 0
        pnl[product] += position[product] * final_mid

    return {
        "total": sum(pnl.values()),
        "by_product": dict(pnl),
        "final_pos": dict(position),
    }


def run_mc_backtest(trader_module, round_num: int, day_num: int,
                    n_paths: int = 30, seed: int = 42,
                    data_dir: Optional[Path] = None) -> dict:
    if data_dir is None:
        data_dir = SCRIPT_DIR / "data"
    reader = DataReader(data_dir)
    price_data = reader.read_prices(round_num, day_num)
    trade_data = reader.read_trades(round_num, day_num)
    timestamps = sorted(price_data.keys())
    if not timestamps:
        return {}
    products = sorted(price_data[timestamps[0]].keys())

    # Fit per-product models
    models = {}
    for product in products:
        prs = [price_data[ts][product] for ts in timestamps
               if product in price_data[ts]]
        mts = []
        for ts in timestamps:
            mts.extend(trade_data.get(ts, {}).get(product, []))
        m = MarketModel(product)
        m.fit(prs, mts, timestamps)
        models[product] = m

    results = []
    for i in range(n_paths):
        rng = random.Random(seed + i)
        results.append(run_one_path(trader_module, models,
                                    timestamps, products, rng))

    totals = [r["total"] for r in results]
    by_prod_agg = defaultdict(list)
    for r in results:
        for p, v in r["by_product"].items():
            by_prod_agg[p].append(v)

    return {
        "n_paths": n_paths,
        "mean_total": statistics.mean(totals),
        "stdev_total": statistics.pstdev(totals) if len(totals) > 1 else 0.0,
        "min_total": min(totals),
        "max_total": max(totals),
        "mean_by_product": {p: statistics.mean(vs) for p, vs in by_prod_agg.items()},
        "models": {p: {
            "kappa": m.kappa, "anchor": m.anchor, "sigma": m.sigma,
            "drift": m.drift, "flow_rate": m.flow_rate,
            "flow_side_ratio": m.flow_side_ratio,
            "n_ticks": m.n_ticks, "mid_start": m.mid_start,
        } for p, m in models.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def load_algo(path: str):
    spec = importlib.util.spec_from_file_location("algo_m", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("algo", help="path to algorithm .py file")
    parser.add_argument("--round", dest="round_num", type=int, default=2)
    parser.add_argument("--day", dest="day_num", type=int, default=1)
    parser.add_argument("--paths", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    module = load_algo(args.algo)
    print(f"MC backtest: {args.algo} round={args.round_num} day={args.day_num} paths={args.paths}")
    result = run_mc_backtest(module, args.round_num, args.day_num,
                             n_paths=args.paths, seed=args.seed)
    print()
    print("Fitted models:")
    for prod, m in result["models"].items():
        print(f"  {prod}: mid_start={m['mid_start']:.1f}  drift={m['drift']:.4f}  "
              f"kappa={m['kappa']:.4f}  anchor={m['anchor']}  sigma={m['sigma']:.3f}  "
              f"flow_rate={m['flow_rate']:.4f}  buy_ratio={m['flow_side_ratio']:.2f}")
    print()
    print(f"Profit across {result['n_paths']} paths:")
    print(f"  mean  = {result['mean_total']:>12.1f}")
    print(f"  stdev = {result['stdev_total']:>12.1f}")
    print(f"  min   = {result['min_total']:>12.1f}")
    print(f"  max   = {result['max_total']:>12.1f}")
    print("  per-product mean:")
    for p, v in result["mean_by_product"].items():
        print(f"    {p}: {v:>12.1f}")


if __name__ == "__main__":
    main()
