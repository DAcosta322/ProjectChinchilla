"""Sweep HYD trend params."""
ALGO = "algorithms/round_3_voucher.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "HydrogelParams.TREND_THRESH":    [0.0, 0.005, 0.01, 0.02, 0.05],
    "HydrogelParams.TREND_MIN_AGE":   [50, 200, 500],
    "HydrogelParams.TREND_FULL_AGE":  [1000, 2000],
    "HydrogelParams.TREND_STRENGTH":  [0.0, 0.3, 0.6, 1.0],
}

def SKIP(p):
    if p["HydrogelParams.TREND_THRESH"] == 0 and p["HydrogelParams.TREND_STRENGTH"] > 0:
        return True
    if p["HydrogelParams.TREND_THRESH"] > 0 and p["HydrogelParams.TREND_STRENGTH"] == 0:
        return True
    if p["HydrogelParams.TREND_FULL_AGE"] <= p["HydrogelParams.TREND_MIN_AGE"]:
        return True
    return False

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
