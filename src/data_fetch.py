import yfinance as yf
import pandas as pd
import os

tickers = ["AAPL", "TSLA", "AMZN", "MSFT", "GOOG"]
start_date = "2023-01-01"
end_date = "2025-10-19"

os.makedirs("data", exist_ok=True)

for ticker in tickers:
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df['Ticker'] = ticker
    df.reset_index(inplace=True)  # Date devient une colonne normale
    df.to_csv(f"data/{ticker}.csv", index=False)  # supprime l'index fantôme
    print(f"{ticker} téléchargé avec {df.shape[0]} lignes")
