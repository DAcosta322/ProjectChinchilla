import pandas as pd

class ProductStats:
    def __init__(self, product_name: str, data: pd.DataFrame):
        self.product_name = product_name
        self.data = data
        self.count = len(data)
        self.mid_price_mean = data["mid_price"].mean() if "mid_price" in data.columns else None
        self.mid_price_std = data["mid_price"].std() if "mid_price" in data.columns else None
        
        # Collect all bid prices from the 3 columns
        bid_cols = [col for col in data.columns if col.startswith('bid_price_')]
        if bid_cols:
            all_bid_prices = data[bid_cols].stack().dropna()
            self.bid_price_mean = all_bid_prices.mean() if not all_bid_prices.empty else None
            self.bid_price_std = all_bid_prices.std() if not all_bid_prices.empty else None
        else:
            self.bid_price_mean = None
            self.bid_price_std = None
        
        # Collect all ask prices from the 3 columns
        ask_cols = [col for col in data.columns if col.startswith('ask_price_')]
        if ask_cols:
            all_ask_prices = data[ask_cols].stack().dropna()
            self.ask_price_mean = all_ask_prices.mean() if not all_ask_prices.empty else None
            self.ask_price_std = all_ask_prices.std() if not all_ask_prices.empty else None
        else:
            self.ask_price_mean = None
            self.ask_price_std = None

    def as_dict(self):
        return {
            "product_name": self.product_name,
            "count": self.count,
            "mid_price_mean": self.mid_price_mean,
            "mid_price_std": self.mid_price_std,
            "bid_price_mean": self.bid_price_mean,
            "bid_price_std": self.bid_price_std,
            "ask_price_mean": self.ask_price_mean,
            "ask_price_std": self.ask_price_std,
        }

    def __str__(self):
        return (
            f"{self.product_name}: count={self.count}, "
            f"mid_mean={self.mid_price_mean:.4f} mid_std={self.mid_price_std:.4f}, "
            f"bid_mean={self.bid_price_mean:.4f} bid_std={self.bid_price_std:.4f}, "
            f"ask_mean={self.ask_price_mean:.4f} ask_std={self.ask_price_std:.4f}"
        )


def load_and_compute_stats(csv_path: str):
    df = pd.read_csv(csv_path, delimiter=";")
    if "product" not in df.columns:
        raise ValueError("Expected 'product' column in the CSV")

    stats_by_product = {}
    for product_name, group in df.groupby("product"):
        stats_by_product[product_name] = ProductStats(product_name, group)

    return stats_by_product


if __name__ == "__main__":
    csv_path = "TUTORIAL_ROUND_1/prices_round_0_day_-1.csv"
    stats = load_and_compute_stats(csv_path)

    for p_name, p_stats in stats.items():
        print(p_stats)
