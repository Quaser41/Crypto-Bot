"""Simple test for live price fetching."""

from data_fetcher import fetch_live_price


def main():
    price = fetch_live_price("BTC")
    print({"BTC": price})


if __name__ == "__main__":
    main()

