"""Dummy trader: no orders, just observes the market."""

from datamodel import TradingState
import json


class Trader:

    def run(self, state: TradingState):
        return {}, 0, ""