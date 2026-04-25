# Round 3 strategy research — 10-strategy sweep

Baseline: `algorithms/round_3_eff4.py` at +$148,312 (4-day backtest)

## Results table

| ID | Strategy | Result | Δ vs eff4 | Verdict |
|---|---|---:|---:|---|
| r1 | Order book imbalance bias | +143,575 | −$4,737 | DROP — adds noise |
| r2 | Trade-tape (aggressor flow) | +128,018 | −$20,294 | DROP — flow signal too noisy |
| r3 | HYD↔VEL pairs / cointegration | n/a | — | DROP — return-corr ≈ 0.01 (no signal exists) |
| r4 | Kalman-filtered anchor | +29,313 | −$119K | DROP (mistuned, would need Q/R search) |
| r5 | Adverse-selection (toxicity) damper | +148,278 | −$35 | KEEP wired but disabled (def TOX_FULL_OFF=0) |
| r6 | Layered passive quoting | +148,009 | −$303 | DROP — neutral, no fill benefit in BT |
| r8 | HYD vol-adaptive MR_STRENGTH | **+152,107** | **+$3,795** | **KEEP — real edge** |
| r9 | L1 reversion bias (bid-ask bounce) | +127,515 | −$20,797 | DROP — signal too noisy at this scale |
| r7 | IV-based fair for OTM vouchers | (skipped) | — | known dead from prior tests (TV instability) |
| r10 | Cross-strike voucher arb | (skipped) | — | known dead — book parity always enforced |

## Diagnostics that led to drops

- **r3 pairs**: ΔHYD vs ΔVELVET return-correlation per day: 0.011, 0.012, −0.005, −0.008. Level correlation unstable (−0.22 to +0.50). No co-movement to exploit.
- **r9 lag1 motivation**: L1 autocorr of returns = −0.12 to −0.17 (bid-ask bounce), but signal too small (<1 tick) to overcome spread cost.
- **r1 imbalance**, **r2 tape**: add directional bias on top of already-tuned MR; over-positions at wrong times.

## Tier 1 + Tier 2 academic-strategy sweep (round 2)

| ID | Strategy | Result | Verdict |
|---|---|---:|---|
| t1 | Avellaneda-Stoikov optimal MM | +$94K (best) to −$10M | DROP — closed-form formula too sensitive to gamma/kappa, blows up |
| **t2** | **Cont-Stoikov OFI (level-1 Δvolumes)** | **+$157,591** | **KEEP — +$3,821 / 4-day** |
| t3 | Cartea-Jaimungal (CJ stoch-vol) | already in vol-scaling | n/a |
| t4 | Glosten-Milgrom adverse selection | skipped (≈ toxicity damper) | n/a |
| t5 | Hawkes trade arrival intensity | tied with OFI baseline | DROP — never spikes 2x |
| t6 | Garman-Klass / Parkinson vol | skipped (we don't have OHLC) | n/a |
| t7 | PCA basket trading | skipped — return-corr ≈ 0 | n/a |
| t8 | OU explicit MR with Z-boost | uniformly negative | DROP — bigger extremes = bigger losses |
| t9 | Kelly criterion sizing | equivalent to existing INV_SKEW | n/a |

**OFI mechanics:** Cont-Stoikov order-flow imbalance uses *changes* in level-1
bid/ask volumes (not just ratios) as a leading indicator:

  e_n = I(P_b_n>=P_b_n-1)*q_b_n - I(P_b_n<=P_b_n-1)*q_b_n-1
      - I(P_a_n<=P_a_n-1)*q_a_n + I(P_a_n>=P_a_n-1)*q_a_n-1
  expected_dprice ~ lambda * EWMA(e_n)

Smoothed over 20 ticks, gain=1.0 added to MR target. Worked where r1 (naive
imbalance) failed because OFI captures *flow direction* via volume changes,
not just current static imbalance.

## Final consolidated algorithm

`algorithms/ROUND_3/round_3_final.py` — eff4 + HYD vol-adaptive MR_STRENGTH +
VEV own-MR + Cont-Stoikov OFI + toxicity damper code (disabled by default).
Result: **+$157,591** (4-day backtest).

Per-day:
- D0: +52,865  (HYD +17.2K, VEL +28K, VEV4/4.5 +8.3K, VEV5K −0.6K)
- D1: +56,247  (HYD +14.5K, VEL +29.6K, VEV4/4.5 +10.6K, VEV5K +1.5K)
- D2: +39,957  (HYD +12.3K, VEL +11.5K, VEV4/4.5 +13.9K, VEV5K +2.2K)
- D3: +4,179   (platform replay)

## Noise stability

Sensitivity sweep (±20% perturbations on key params):
- PROFIT_DIST: ±$5K range
- VOL_REF: −$11K at +20% (asymmetric, current value near optimal)
- VOL_SPAN: ±$2K range
- TV_EWMA_SPAN: ±$3K range (50 is local optimum)
- MR_STRENGTH: −$11K at +20% (current value near optimal)

Total stays in +$140K to +$153K band across all perturbations — robust.

Hardening:
- Returns clipped to ±50 before var update (prevents jump-blowup)
- Var EWMA initialized to VOL_REF² (no warm-up artifacts)
- vol_factor clipped to [0.5, 2.0]
- TV estimate clipped to ±50
- profit_decay monotone (no oscillation at threshold)
- Quote prices clipped against fv_eff (never post worse than valuation)
- VEV_5000 has 100-sample warm-up before first trade
