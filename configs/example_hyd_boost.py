"""Example sweep config: HYDROGEL_PACK boost grid on round 3 days 0-2.

Run:
    python sweep.py configs/example_hyd_boost.py --top 10
"""

ALGO = "algorithms/round_3_voucher.py"
ROUND = 3
DAYS = [0, 1, 2]
MODE = "bt"   # "bt" = deterministic backtester, "mc" = monte-carlo

GRID = {
    "HydrogelParams.BOOST_THRESHOLD": [10, 20, 30, 40, 50],
    "HydrogelParams.BOOST_GAIN":      [1, 3, 5, 10, 20],
}

TRACK = ["HYDROGEL_PACK"]
