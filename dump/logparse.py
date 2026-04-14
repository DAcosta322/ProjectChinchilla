import json
from pathlib import Path
#usage: Put this in the folder for all your submission feedback files, then change "dump" to folder name in dd.
#enter the sbmission id to see some data.
subnum = input("Enter submission id")

dd = "dump/" + subnum
jj = subnum + ".json"
ll = subnum + ".log"
d = Path(dd)
j = json.loads((d/jj).read_text())
log = json.loads((d/ll).read_text())
print("Submission", subnum)
print("profit", j["profit"])
print("positions", j["positions"])
print("activities rows", len(j["activitiesLog"].splitlines()))

# trade history stats
trade = log["tradeHistory"]
print("trades", len(trade))
print("symbols", {s: sum(1 for t in trade if t["symbol"]==s) for s in {"TOMATOES","EMERALDS"}})