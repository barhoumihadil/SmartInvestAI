import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px

FEATURES = ['Return', 'SMA_7', 'EMA_7', 'Volatility_7']

def plot_histograms(df, color="#1f77b4"):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for i, f in enumerate(FEATURES):
        axes[i].hist(df[f], bins=30, color=color, edgecolor='black')
        axes[i].set_title(f'Distribution de {f}')
    plt.tight_layout()
    return fig

def plot_boxplots(df, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(12,6))
    df[FEATURES].plot(kind='box', ax=ax, color=dict(boxes=color, whiskers=color, caps=color, medians=color))
    ax.set_title("Boxplots des features")
    return fig

def plot_heatmap(df, cmap="coolwarm"):
    features = FEATURES + ['Target']
    corr = df[features].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap=cmap, ax=ax)
    ax.set_title("Matrice de corrélation")
    return fig

def plot_price_evolution(df, color="#1f77b4"):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['Date'], df['Adj Close'], label='Adj Close', color=color)
    ax.plot(df['Date'], df['SMA_7'], label='SMA 7', color='orange')
    ax.plot(df['Date'], df['EMA_7'], label='EMA 7', color='green')
    ax.fill_between(df['Date'],
                    df['Adj Close'] - df['Volatility_7'],
                    df['Adj Close'] + df['Volatility_7'],
                    color='gray', alpha=0.2, label='Volatilité ±1')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix")
    ax.set_title("Prix et indicateurs")
    ax.legend()
    return fig

def plot_bar(df, column):
    if column not in FEATURES:
        raise ValueError(f"Colonne invalide pour bar plot. Choisir parmi {FEATURES}")
    fig = px.bar(df, x=df.index, y=column, title=f"Bar plot de {column}")
    return fig
