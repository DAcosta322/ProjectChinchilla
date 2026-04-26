"""Sweep bot-flow signal gains on round 4 days 1-3.

Round 4 adds counterparty (Mark XX) identities to the trade CSVs.
Round 3 trades have empty buyer/seller, so signal would be flat there.
Baseline (all gains 0) reproduces round_3_voucher exactly.

Run:
    python sweep.py configs/botflow_sweep.py --top 15
"""

ALGO = "algorithms/round_4_botflow.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    # Mark 01 vs Mark 22 voucher pair gain (per voucher).
    "BotFlowParams.M01_M22_GAIN": [0.0, 0.05, 0.1, 0.2, 0.5, 1.0],
    # Mark 67 vs Mark 49 VELVET pair gain.
    "BotFlowParams.M67_M49_GAIN": [0.0, 0.05, 0.1, 0.2, 0.5],
    # Decay half-life in ticks.
    "BotFlowParams.HALF_LIFE":     [100, 300],
}

TRACK = ["VELVETFRUIT_EXTRACT", "VEV_5500", "VEV_6000"]
