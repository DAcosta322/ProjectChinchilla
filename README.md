# IMC Prosperity 4

Algorithmic team: Daniel Acosta and Ruiyang Ye

Manual team: Daniel Singh and Roy Sulaiman.

## Who are we

First year students at UWaterloo (all BMath), first time doing this.
Our team was Burrito Warriors, top 11% global for algorithmic, top 62% global for manual, and top 25 in overall Canada.

## Methodology for Algorithmic

Being first years, most of us hadn't taken any related stat, acturial science, computer science, etc. courses that would help us with knowing what to do.
We basically tried every strategy we thought could work well of without knowing much of how it would work.
We also tried to not overcomplicate our strategies to get actual good information from our testing and to do better next time.
We developed a backtester so that we didn't have to rely on the slower testing of the official IMC testing (~2min compaired to ~15sec), which is important we it comes to training parameters, would recommend using like Pytorch and just exporting the values.

## Methodology for Manual
 
This writeup covers my approach and results for the four manual rounds in IMC Prosperity 4. I did the bulk of the manual analysis; teammate Ruiyang Ye contributed on Round 3. Algorithmic strategies are documented separately. Daniel Singh contributed with Round 5, and we'll get his input soon
 
## Structural Overview
 
- [Round 1: An Intarian Welcome](#round-1-an-intarian-welcome)
- [Round 2: Invest & Expand](#round-2-invest--expand)
- [Round 3: Celestial Gardeners' Guild](#round-3-celestial-gardeners-guild)
- [Round 4: Exotic Options Pricing](#round-4-exotic-options-pricing)
## Round 1: An Intarian Welcome
 
Two opening auctions for `DRYLAND_FLAX` and `EMBER_MUSHROOM`. Each was a sealed-bid call auction: submit a single limit order, the exchange picks the clearing price that maximises traded volume (tie-broken by higher price), and any inventory traded is bought back at a fixed price by the Merchant Guild: 30 for flax, 20 for mushrooms (with a 0.10/unit fee on mushrooms). Per-unit margin is `(buyback − P_clear)` minus fees. Players submit last, so they sit at the back of the queue at any price level they join.
 
Two exploits drove the problem: a **priority-jump** (bidding strictly above the clearing price to skip the existing queue) and a **tie-break-up trap** (excess volume tying crossed volumes between price levels, dragging `P_clear` up to a worse tier).
 
### DRYLAND_FLAX (buyback 30)
 
| Price | Cum. Bids ≥ P | Cum. Asks ≤ P | Crossed |
|-------|---------------|---------------|---------|
| 28    | 47,000        | 40,000        | 40,000  |
| 29    | 35,000        | 40,000        | 35,000  |
| 30    | 30,000        | 40,000        | 30,000  |
 
Natural `P_clear` = 28. Bidding at 28 sits behind 47,000 priority bids for 40,000 asks, so the fill is zero. The fill ceiling at any profitable price was hard-capped at 10,000 units (40k asks minus 30k pre-existing bids at 30).
 
Cliff edges:
- Bid 29 with V ≥ 5,000 ties crossed volume at 29 with 28, dragging `P_clear` to 29 (margin halves).
- Bid 30 or 31 with V ≥ 10,000 ties 30 with 28 and 29, dragging `P_clear` to 30 (margin → 0).
The optimal play was to *intentionally* trigger the tie at 29 by bidding 9,999 at 31, sacrificing 1 unit of margin to multiply fill from 4,999 to 9,999. Submitted **BUY 9,999 at 31**, profit **9,999**.
 
### EMBER_MUSHROOM (buyback 20, fee 0.10/unit)
 
Margin was `19.90 − P_clear`. Pre-auction book had 86,000 crossed volume at P=15 with a flat ask plateau of 91,000 from P=16 through P=18.
 
Same logic, different cliffs:
- Bidding V ≥ 5,000 at P ≥ 16 ties crossed volume at 16 with 15, pushing `P_clear` up to 16. This *helps*, since it opens 5,000 additional asks.
- Bidding 20,000+ at 17 ties 17 with 16 (because there are exactly 0 asks at 17, the ask plateau is flat from 16 to 17), pushing `P_clear` to 17 and crushing margin from 3.90 to 2.90. Volume must be capped at 19,999.
Submitted **BUY 19,999 at 17** (any price between 17 and 20 yields identical fill, since the priority-jump is binary), profit **77,996.10**.
 
Both products were hit at the optimum. Round 1 was purely mechanical, with no behavioural component, so once the cliff edges and queue priority were mapped out, the optimum was forced.
 
## Round 2: Invest & Expand
 
Each team allocated integer points across three pillars under `r + s + v ≤ 100`:
 
- Research: `R(r) = 200,000 · ln(1+r) / ln(101)`
- Scale: `S(s) = 7s / 100`
- Speed: competitive multiplier ranked against the full 15,000-team field, dense-rank-with-skip. `m(v) = 0.9 − 0.8 · (rank − 1) / (N − 1)`, bounded in [0.1, 0.9].
PnL = `R(r) · S(s) · m(v) − 500(r + s + v)`.
 
The mathematical optimum under a uniform-field assumption peaks at `v = 35` with `(r, s) = (16, 49)`, expected PnL ≈ 112k. Because every quantitative team finds 35, it becomes a focal point. Stepping up by one to overcut the cluster is only profitable if `P(V = k+1) > 2.55% · m(k)`.
 
I built a behavioural prior from a 20,290-message Discord scrape, which showed a bimodal field: ~28% at v=0 (safety/qualifier-anxious players), a geometric cascade through v=1, 2, 3… with `ρ ≈ 0.65`, and a long-tail spike at v=100 (sabotage). Applying the break-even rule to this prior, the cascade halted at **v = 6** where the marginal mass at v=7 (~0.9%) failed to clear the threshold (~1.02%).
 
Submitted **(22, 72, 6)**. Actual winning speed was **v = 42**.
 
| v   | Optimal (r, s) | Base Product | Realised Multiplier | Expected PnL |
|-----|----------------|--------------|---------------------|--------------|
| 6   | (22, 72)       | 684,878      | ~0.58 (modelled)    | ~345,586     |
| 36  | (16, 48)       | 412,500      | ~0.50 (50th %ile)   | ~156,250     |
| 46  | (14, 40)       | 328,600      | ~0.78 (85th %ile)   | ~206,308     |
| 53  | (13, 34)       | 272,200      | ~0.86 (95th %ile)   | ~184,092     |
 
The Discord-driven model massively overestimated my realised multiplier at v=6. The model assumed the field stayed bottom-anchored, projecting I'd land near the 60th percentile. In reality the bulk of the field was in the 30s–40s, and v=6 fell into the dead zone between the v=0 cluster and the math-optimum mountain. Realised multiplier was substantially lower than projected.
 
Three errors caused this:
 
1. **Selection bias on the Discord scrape.** The 57 mentions of v=0 were the loud minority. The silent majority of quants were computing v=35 privately. The Discord prior optimised against a field that didn't materialise.
2. **Micro-steps vs. macro-leaps.** My break-even rule assumed players make rational +1 overcuts. Humans under cognitive load jump to round numbers (10, 20, 30, 40), which are Schelling points. I modelled none of these.
3. **Greedy search.** My algorithm walked up from 0 in +1 steps and halted on the first thin region. This forbids crossing an empty valley to find a higher peak. The math optimum cluster at 35 was on the other side of the valley, and the winning macro-leap to 42–46 was past that.
## Round 3: Celestial Gardeners' Guild
 
A two-bid reverse auction. Submit `(b_1, b_2)`. Counterparty reserves were uniformly distributed on the integer grid `{670, 675, …, 920}` (51 tiers, increment 5). Resale price = 920.
 
- If `b_1 > reserve`, trade at `b_1`.
- Else if `b_2 > reserve`, trade at `b_2`. If `b_2 ≤ avg_b_2` (global mean of all teams' second bids), profit penalised by `((920 − avg_b_2) / (920 − b_2))^3`.
Since bids must strictly exceed reserve and reserves sit on a 5-grid, the cheapest bid clearing exactly `k` tiers is `b(k) = 666 + 5k`. Let `k_1, k_2` be tiers cleared.
 
Total unpenalised profit: `P(k_1, k_2) = k_1(254 − 5k_1) + (k_2 − k_1)(254 − 5k_2)`.
 
Partial derivatives → `k_2 = 2k_1` and `254 − 15k_1 = 0`, giving `k_1 = 16.93`, `k_2 = 33.87`. Discretised: `k_1 = 17, k_2 = 34`, so `b_1 = 751`, `b_2 = 836`. The Hessian eigenvalues are −5 and −15 (strictly concave), so `(751, 836)` is the genuine unpenalised global maximum at 4,301 profit per 51-tier unit.
 
The penalty is mean-based, not rank-based. Solving the breakeven for pivoting to `b_2 = 841`: the mean must drift above 836.118 for the safe harbour to crack. With thousands of casual bidders anchoring round numbers (700, 750, 800) and Level-0 quants solving the single-variable optimum (`b_2 = 791`, the parabola peak when the `b_1` interaction is ignored), my model forecast the mean landing in 810–825.
 
Ruiyang Ye disagreed. Carrying forward Round 2's lesson, that the math optimum is the focal trap rather than the answer, he proposed a partial macro-leap to **(796, 856)**: ~9 tier-steps above the math optimum on `b_1` and ~4 above on `b_2`, anticipating that the realised mean would exceed 836 once the rest of the field also tried to clear the safe-harbour level.
 
Team submitted **(796, 856)**. Actual optimal was **(761, 861)**.
 
The structural call to override the math optimum was correct, but the calibration was off on both bids:
 
- `b_1 = 796` was too aggressive. Each first-bid trade gave up 35 in margin vs the optimum's 761. Across 26 trades cleared by `b_1`, that's ~900 of forgone unpenalised profit.
- `b_2 = 856` was 5 short of the actual optimum at 861. Assuming the realised mean landed somewhere between 836 and 855, `b_2 = 856` likely cleared the penalty (or took a minor one), but undersized the leap relative to the field's true centre of gravity.
Result was meaningfully better than what (751, 836) would have produced, since the math optimum would have eaten the penalty, but ~5% below the realised optimum. The directional thesis was right; the magnitude was wrong. Calibrating the size of the macro-leap is the part neither Round 2 nor Round 3 taught me how to do reliably.
 
## Round 4: Exotic Options Pricing
 
Twelve derivatives on a simulated underlying `S` following zero-drift GBM with `S_0 = 50`, annualised `σ = 2.51` (extreme), 4 steps/day, 252 trading days/year, `Δt = 1/1008`. Two-week expiry = 40 grid steps; three-week expiry = 60 steps.
 
| Contract  | Kind          | Spec                | Expiry | Bid    | Ask    | Cap |
|-----------|---------------|---------------------|--------|--------|--------|-----|
| AC        | Spot          | N/A                 | N/A    | 49.975 | 50.025 | 200 |
| AC_35_P   | European put  | K = 35              | 3w     | 4.33   | 4.35   | 50  |
| AC_40_P   | European put  | K = 40              | 3w     | 6.50   | 6.55   | 50  |
| AC_45_P   | European put  | K = 45              | 3w     | 9.05   | 9.10   | 50  |
| AC_50_P   | European put  | K = 50              | 3w     | 12.00  | 12.05  | 50  |
| AC_50_C   | European call | K = 50              | 3w     | 12.00  | 12.05  | 50  |
| AC_60_C   | European call | K = 60              | 3w     | 8.80   | 8.85   | 50  |
| AC_50_P_2 | European put  | K = 50              | 2w     | 9.70   | 9.75   | 50  |
| AC_50_C_2 | European call | K = 50              | 2w     | 9.70   | 9.75   | 50  |
| AC_50_CO  | Auto-chooser  | K = 50, choose @2w  | 3w     | 22.20  | 22.30  | 50  |
| AC_40_BP  | Binary put    | pays 10 if S_T < 40 | 3w     | 5.00   | 5.10   | 50  |
| AC_45_KO  | Discrete DOP  | K = 45, B = 35      | 3w     | 0.150  | 0.175  | 500 |
 
Scoring: mean PnL over 100 fixed paths, multiplied by contract size 3,000.
 
Under `r = q = 0`, all vanillas reduce to Black-Scholes. The chooser decomposes exactly as `Call(K, T_expiry) + Put(K, T_choice)` (Rubinstein 1991; Hull §26.6); the auto-pick-ITM rule is equivalent to holding both legs at zero rates, since the OTM leg expires worthless and was never paid for separately. The binary put reduces to `payout · Φ(−d_2)`.
 
The discrete down-and-out put was the difficult contract. Three approaches were on the table: the Broadie-Glasserman-Kou continuity correction (`B_eff = B · exp(−β · σ · √Δt)`, β ≈ 0.5826), Reiner-Rubinstein continuous closed form with a discretisation bound, and direct Monte Carlo with antithetic variates and a Brownian-bridge survival-probability control variate.
 
I used MC + Brownian-bridge CV, achieving ~63% variance reduction. My fair value for AC_45_KO came in at **0.207**, against market quote bid 0.150 / ask 0.175, implying a BUY edge of ~0.032 per unit and ~47k expected PnL at the 500-unit cap.
 
Submitted portfolio:
 
| Trade            | Edge/Unit | Volume | Expected PnL |
|------------------|-----------|--------|--------------|
| SELL AC_50_CO    | +0.302    | 50     | +45,351      |
| SELL AC_40_BP    | +0.232    | 50     | +34,808      |
| BUY AC_45_KO     | +0.032    | 500    | +47,426      |
| BUY AC_50_P_2    | +0.121    | 50     | +18,106      |
| BUY AC_50_C_2    | +0.121    | 50     | +18,106      |
| **Total**        |           |        | **+163,797** |
 
The published optimal was:
 
```
BUY  160 AC
BUY  50  AC_50_P     T+21
BUY  50  AC_35_P     T+21
BUY  50  AC_40_P     T+21
BUY  50  AC_45_P     T+21
SELL 50  AC_60_C     T+21
BUY  50  AC_50_P_2   T+14
SELL 50  AC_50_C_2   T+14
SELL 50  AC_50_CO    T+14/21
SELL 50  AC_40_BP    T+21
SELL 500 AC_45_KO    T+21
Profit: 177,980
```
 
Matched on `SELL AC_50_CO`, `SELL AC_40_BP`, and `BUY AC_50_P_2`. The expensive miss was the KO put: I bought 500, optimal sold 500. Fair value was wrong in direction.
 
At σ = 2.51 over 3 weeks, the drift correction `−½σ²T ≈ −0.094` puts the median of `S_T` around 45.5, but the standard deviation of `ln S_T` is ~0.612, which is enormous. Over 60 monitoring points against a barrier 30% below spot in a regime with ~16% daily moves, the realised knockout probability on simulation was 87–92%. At those rates the discrete KO put is worth roughly 0.10–0.12, not 0.207. The market quote of 0.150 was already above fair value; it was a SELL.
 
The Brownian-bridge CV construction was correct, but I used the Reiner-Rubinstein *continuous*-monitoring DOP as the expected value of the CV, which overestimates survival probability under discrete monitoring in this regime. The BGK correction inward-adjusts the barrier by ~0.5826 · σ · √Δt ≈ 4.6%, non-trivial at this volatility. I cited BGK 1997 explicitly in my literature review prompt and still underweighted it. The miss on AC_50_C_2 (I bought, optimal sold) followed from the same model error: the chooser-vs-vanilla hedge was correct in shape but inverted in one leg.
 
Loss versus optimal was ~14k of the realised optimum's 178k.
 
## Conclusion
 
Round 1 was purely mechanical; the math forced the optimum. Round 2 was field-dependent and I trusted a biased Discord prior, landing in the dead zone between the casual cluster at v=0 and the quant cluster at v=35. Round 3 was field-dependent and Ruiyang correctly overrode the math optimum, but the macro-leap was miscalibrated: too aggressive on `b_1`, slightly short on `b_2`. Round 4 was model-dependent, and the loss came from underweighting a known discretisation correction in an extreme-volatility regime.
 
The unifying mistake across Rounds 2–4 was treating the math optimum as the answer rather than as a focal point, except in Round 3, where the override was made but mis-sized. The math optimum tells you where the cluster forms; the winning play is usually one calibrated step past it. Calibrating *one step*, and not more, is the hard part.
 
On tooling: AI assistants are useful for the mechanical components (closed-form derivations, calculus, MC variance reduction, optimisation sweeps) and weaker on the judgement components (modelling humans, calibrating risk appetite, recognising when an empirical signal contradicts the model). Every round I lost, the model produced a defensible answer and I shipped it without enough adversarial pressure of my own.
