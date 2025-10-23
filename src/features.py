import pandas as pd
import glob

def add_features(df):
    df = df.copy()
    
    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'
    
    df[price_col] = pd.to_numeric(df[price_col], errors='coerce')

    df['Return'] = df[price_col].pct_change(fill_method=None)

    df['SMA_7'] = df[price_col].rolling(7).mean()
    df['EMA_7'] = df[price_col].ewm(span=7, adjust=False).mean()
    df['Volatility_7'] = df['Return'].rolling(7).std()
  
    df['Future_Close'] = df.groupby('Ticker')[price_col].shift(-7)
    df['Target'] = (df['Future_Close'] > df[price_col]).astype(int)
    
    df = df.dropna()
    return df

def prepare_dataset():
    all_files = glob.glob("data/*.csv")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    df_feat = add_features(df)
    return df_feat

if __name__ == "__main__":
    df = prepare_dataset()
    print(df.head())
