import os
import numpy as np
import pandas as pd
import yfinance as yf

TICKER = os.getenv("STOCK_TICKER", "AAPL")


def fetch_data(ticker):
  print(f"Fetching data for {ticker}...")
  df = yf.download(ticker, period="max", interval="1d", progress=False)

  if df.empty:
    raise ValueError(f"No data found for ticker: {ticker}")

  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df[["Close"]].dropna()
  return df


def run_backtest(df):
  df["SMA_50_Daily"] = df["Close"].rolling(window=50).mean()
  df["SMA_200_Daily"] = df["Close"].rolling(window=200).mean()

  df_weekly = df["Close"].resample("W").last().to_frame()
  df_weekly["SMA_50_Weekly"] = df_weekly["Close"].rolling(window=50).mean()
  df_weekly["SMA_200_Weekly"] = df_weekly["Close"].rolling(window=200).mean()

  df = df.join(
      df_weekly[["SMA_50_Weekly", "SMA_200_Weekly"]], how="left"
  ).ffill()

  df["Trigger_50D"] = (df["Close"] < df["SMA_50_Daily"]) & (
      df["Close"].shift(1) >= df["SMA_50_Daily"].shift(1)
  )
  df["Trigger_200D"] = (df["Close"] < df["SMA_200_Daily"]) & (
      df["Close"].shift(1) >= df["SMA_200_Daily"].shift(1)
  )
  df["Trigger_50W"] = (df["Close"] < df["SMA_50_Weekly"]) & (
      df["Close"].shift(1) >= df["SMA_50_Weekly"].shift(1)
  )
  df["Trigger_200W"] = (df["Close"] < df["SMA_200_Weekly"]) & (
      df["Close"].shift(1) >= df["SMA_200_Weekly"].shift(1)
  )

  df["Total_Buys_Today"] = (
      df["Trigger_50D"].astype(int)
      + df["Trigger_200D"].astype(int)
      + df["Trigger_50W"].astype(int)
      + df["Trigger_200W"].astype(int)
  )

  return df


def calculate_performance(df, years):
  end_date = df.index[-1]
  start_date = end_date - pd.DateOffset(years=years)

  sub_df = df.loc[df.index >= start_date].copy()
  if sub_df.empty:
    return None

  shares_held = 0
  total_invested = 0.0

  for _, row in sub_df.iterrows():
    buys = int(row["Total_Buys_Today"])
    price = float(row["Close"])
    if buys > 0:
      shares_held += buys
      total_invested += buys * price

  current_price = float(df["Close"].iloc[-1])
  portfolio_value = shares_held * current_price
  net_profit = portfolio_value - total_invested
  absolute_return_pct = (
      (net_profit / total_invested * 100) if total_invested > 0 else 0.0
  )
  annualized_return_pct = (
      ((portfolio_value / total_invested) ** (1 / years) - 1) * 100
      if total_invested > 0 and years > 0
      else 0.0
  )

  return {
      "horizon": f"{years} Year" if years > 1 else "1 Year",
      "shares_held": shares_held,
      "total_invested": round(total_invested, 2),
      "portfolio_value": round(portfolio_value, 2),
      "net_profit": round(net_profit, 2),
      "absolute_return_pct": round(absolute_return_pct, 2),
      "annualized_return_pct": round(annualized_return_pct, 2),
  }


def generate_markdown_report(ticker, current_price, metrics_list):
  md = f"""# 📈 Moving Average Accumulation Strategy Report: `{ticker.upper()}`

> **Strategy Rule:** Buy 1 unit whenever the price crosses below the 50-day, 200-day, 50-week, or 200-week moving averages.

* **Current Market Price:** `${current_price:,.2f}`

### Performance Summary

| Time Horizon | Shares Held | Invested ($) | Final Value ($) | Absolute Return | Annualized Return |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""

  for m in metrics_list:
    abs_emoji = "🟢" if m["absolute_return_pct"] >= 0 else "🔴"
    ann_emoji = "🟢" if m["annualized_return_pct"] >= 0 else "🔴"
    md += f"| **{m['horizon']}** | {m['shares_held']} | ${m['total_invested']:,.2f} | ${m['portfolio_value']:,.2f} | {abs_emoji} {m['absolute_return_pct']:+.2f}% | {ann_emoji} {m['annualized_return_pct']:+.2f}% |\n"

  md += "\n*Report automatically generated via GitHub Actions & Python.*"
  return md


if __name__ == "__main__":
  df = fetch_data(TICKER)
  df_backtested = run_backtest(df)
  current_price = float(df["Close"].iloc[-1])

  horizons = [1, 5, 10]
  metrics_list = []
  for h in horizons:
    res = calculate_performance(df_backtested, h)
    if res:
      metrics_list.append(res)

  report_md = generate_markdown_report(TICKER, current_price, metrics_list)

  # Write directly to GitHub Actions Job Summary
  github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
  if github_summary_path:
    with open(github_summary_path, "a") as summary_file:
      summary_file.write(report_md)
    print("Report successfully published to GitHub Actions Job Summary!")
  else:
    print(report_md)
