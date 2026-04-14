import pandas as pd
const = {'symbol': ['EMERALDS','TOMATOES'], 
        'mean': [10000, 5000]}
constdf = pd.DataFrame(const)

class ProductStats:
    def __init__(self, product_name: str, data: pd.DataFrame):
        self.symbol = product_name
        self.data = data
        self.price = data["price"] if "price" in data.columns else None
        self.quantity = data["quantity"] if "quantity" in data.columns else None
        self.netqty = data["quantity"].sum()
        self.trade_count = len(data)
    def as_dict(self):
        return {
            "product_name": self.symbol,
            "price": self.price,
            "quantity": self.quantity,
            "netquantity": self.netqty,
            "trade_count": self.trade_count,
        }

    def __str__(self):
        return (
            f"{self.symbol}: NetQty={self.netqty}, TradeCount={self.trade_count}"
            #f"buy_mean={self.mid_price_mean:.4f} mid_std={self.mid_price_std:.4f}, "
            #f"bid_mean={self.bid_price_mean:.4f} bid_std={self.bid_price_std:.4f}, "
        )


def load_and_compute_stats(csv_path: str):
    df = pd.read_csv(csv_path, delimiter=";")
    if "symbol" not in df.columns:
        raise ValueError("Expected 'symbol' column in the CSV")

    stats_by_product = {}
    for symbol, group in df.groupby("symbol"):
        stats_by_product[symbol] = ProductStats(symbol, group) # type: ignore

    return stats_by_product


if __name__ == "__main__":
    csv_path = "TUTORIAL_ROUND_1/trades_round_0_day_-1.csv"
    stats = load_and_compute_stats(csv_path)

    for p_name, p_stats in stats.items():
        print(p_stats)
