"""Sweep 'lay most of the signal on the bots' — push signal influence harder.

Three knobs control bot dominance:
  CONVICTION_AMP : amplifies |conv| before weight (blend mode)
  COMBINE_MODE   : 'blend' or 'override' (signal replaces MR)
  SIGNAL_FULL    : lower => same signal hits ±POS_LIMIT faster
  SIGNAL_THRESH  : 0 = always engage signal
"""

ALGO = "algorithms/round_4_event_aggressive.py"
ROUND = 4
DAYS = [1, 2, 3]
MODE = "bt"

GRID = {
    "BotEventParams.CONVICTION_AMP": [1.0, 2.0, 4.0, 8.0],
    "BotEventParams.SIGNAL_FULL":    [25.0, 40.0, 60.0, 100.0],
    "BotEventParams.SIGNAL_THRESH":  [0.0, 3.0, 6.0, 10.0],
    "BotEventParams.HALF_LIFE":      [50, 100],
    "BotEventParams.COMBINE_MODE":   ["blend", "override"],
    "BotEventParams.AT_MID_WEIGHT":  [1.0],
    "BotEventParams.M14_WEIGHT":     [0.1],
}

TRACK = ["VELVETFRUIT_EXTRACT"]
