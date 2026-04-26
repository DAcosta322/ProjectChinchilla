"""Sweep config: HYD/VEL active-flatten - try VERY small gains."""
ALGO = "algorithms/round_3_voucher.py"
ROUND = 3
DAYS = [0, 1, 2]
MODE = "bt"

GRID = {
    "HydrogelParams.NEUTRAL_BAND": [0, 5, 10, 20],
    "HydrogelParams.NEUTRAL_GAIN": [0.0, 0.05, 0.1, 0.2, 0.5],
    "VelvetParams.NEUTRAL_BAND":   [0, 3, 5],
    "VelvetParams.NEUTRAL_GAIN":   [0.0, 0.05, 0.1, 0.2, 0.5],
}

def SKIP(p):
    if p["HydrogelParams.NEUTRAL_BAND"] == 0 and p["HydrogelParams.NEUTRAL_GAIN"] > 0:
        return True
    if p["VelvetParams.NEUTRAL_BAND"] == 0 and p["VelvetParams.NEUTRAL_GAIN"] > 0:
        return True
    if p["HydrogelParams.NEUTRAL_BAND"] > 0 and p["HydrogelParams.NEUTRAL_GAIN"] == 0:
        return True
    if p["VelvetParams.NEUTRAL_BAND"] > 0 and p["VelvetParams.NEUTRAL_GAIN"] == 0:
        return True
    return False

TRACK = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT"]
