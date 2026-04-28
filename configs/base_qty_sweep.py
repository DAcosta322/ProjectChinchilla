"""Sweep BASE_QTY (passive quote size) on HYD + VEL to integrate pure-MM
spread capture on top of the signal-driven target.

Hypothesis: pure_mm earned +23K from spread alone with quote size 30. The
event_aggressive algo currently uses BASE_QTY=10 — leaving ~$0-23K of spread
income unclaimed when position is below limit.
"""

ALGO = "algorithms/round_4_event_aggressive.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "HydrogelParams.BASE_QTY":   [10, 20, 30, 50],
    "VelvetParams.BASE_QTY":     [10, 20, 30, 50],
}

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
