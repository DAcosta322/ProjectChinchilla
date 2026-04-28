"""Push trend-follow harder on HYD and VEL.

Theoretical max DP shows positions should drift to ±limit at end of day; current
TREND_STRENGTH=0.3/0.6 may not be enough to ride sustained drift through anchor.
"""

ALGO = "algorithms/ROUND_4/round_4_max_scale.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "HydrogelParams.TREND_STRENGTH":  [0.3, 0.6, 1.0, 1.5],
    "HydrogelParams.TREND_MIN_AGE":   [20, 50, 100],
    "VelvetParams.TREND_STRENGTH":    [0.6, 1.0, 1.5, 2.0],
    "VelvetParams.TREND_MIN_AGE":     [50, 100, 200],
}

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
