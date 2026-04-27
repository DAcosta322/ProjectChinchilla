"""Sweep the aggressive event-driven VELVET algorithm.

Variables: SIGNAL_FULL (gain), SIGNAL_THRESH (engage point), HALF_LIFE (decay),
AT_MID_WEIGHT (down-weight at-mid prints), M14_WEIGHT (mirror confirm).
"""

ALGO = "algorithms/round_4_event_aggressive.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "BotEventParams.SIGNAL_FULL":   [15.0, 25.0, 40.0, 60.0, 100.0],
    "BotEventParams.SIGNAL_THRESH": [3.0, 6.0, 10.0],
    "BotEventParams.HALF_LIFE":     [25, 50, 100],
    "BotEventParams.AT_MID_WEIGHT": [0.0, 0.5, 1.0],
    "BotEventParams.M14_WEIGHT":    [0.0, 0.1],
}

TRACK = ["VELVETFRUIT_EXTRACT"]
