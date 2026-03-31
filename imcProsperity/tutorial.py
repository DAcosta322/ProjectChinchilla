from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import pandas as pd

### General ### General ### General ### General ### General ###

COMMODITY_SYMBOL = "EMERALD"
STOCK_SYMBOL = "TOMATO"

POS_LIMITS = {
    COMMODITY_SYMBOL : 80,
    STOCK_SYMBOL : 80
}

### General ### General ### General ### General ### General ###
### Utilities ### Utilities ### Utilities ### Utilities ### Utilities ###

class ProductTrader:
    def __init__(self,name,mean,sd=0,group=None):
        
        self.orders = [] 
        self.price = []
        
        self.mean = mean
        self.name= name
        self.sd = sd
        self.group = name if group is None else group
        pass

    def getPrice(self):
        return
        
    def getSD(self):
        self.sd = pd.DataFrame(self.price).stdev()
        return self.sd

    def getMean(self):
        self.mean = pd.DataFrame(self.price).mean()
        return self.mean


### Utilities ### Utilities ### Utilities ### Utilities ### Utilities ###
### Commodity ### Commodity ### Commodity ### Commodity ### Commodity ###

class CommodityTrader(ProductTrader):
    def __init__(self, name, sd=0):
        super().__init__(COMMODITY_SYMBOL, sd)
   

### Commodity ### Commodity ### Commodity ### Commodity ### Commodity ###
### Stock ### Stock ### Stock ### Stock ### Stock ###

class StockTrader(ProductTrader):
    def __init__(self, name, sd=0, group=None):
        super().__init__(STOCK_SYMBOL, sd, group)
        

### Stock ### Stock ### Stock ### Stock ### Stock ###
### Trader ### Trader ### Trader ### Trader ### Trader ###

class Trader:

    def bid(self):
        return 15
    
    def run(self, state: TradingState):
        """Only method required. It takes all buy and sell orders for all
        symbols as an input, and outputs a list of orders to be sent."""

        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))

        # Orders to be placed on exchange matching engine
        result = {}
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            acceptable_price = 10  # Participant should calculate this value
            print("Acceptable price : " + str(acceptable_price))
            print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
    
            if len(order_depth.sell_orders) != 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                if int(best_ask) < acceptable_price:
                    print("BUY", str(-best_ask_amount) + "x", best_ask)
                    orders.append(Order(product, best_ask, -best_ask_amount))
    
            if len(order_depth.buy_orders) != 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                if int(best_bid) > acceptable_price:
                    print("SELL", str(best_bid_amount) + "x", best_bid)
                    orders.append(Order(product, best_bid, -best_bid_amount))
            
            result[product] = orders
    
        # String value holding Trader state data required. 
        # It will be delivered as TradingState.traderData on next execution.
        traderData = ""
        
        # Sample conversion request. Check more details below. 
        conversions = 0
        return result, conversions, traderData