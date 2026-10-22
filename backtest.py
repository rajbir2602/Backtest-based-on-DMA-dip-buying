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


def run_scenario_backtest(df, trigger_col_name, horizon_years):
  """Runs a backtest for a single specific MA trigger condition over a given horizon."""
  end_date = df.index[-1]
  start_date = end_date - pd.DateOffset(years=horizon_years)

  sub_df = df.loc[df.index >= start_date].copy()
  if sub_df.empty:
    return None

  shares_held = 0
  total_invested = 0.0

  for _, row in sub_df.iterrows():
    if row[trigger_col_name]:
      price = float(row["Close"])
      shares_held += 1
      total_invested += price

  current_price = float(df["Close"].iloc[-1])
  portfolio_value = shares_held * current_price
  net_profit = portfolio_value - total_invested
  absolute_return_pct = (
      (net_profit / total_invested * 100) if total_invested > 0 else 0.0
  )
  annualized_return_pct = (
      ((portfolio_value / total_invested) ** (1 / horizon_years) - 1) * 100
      if total_invested > 0 and horizon_years > 0
      else 0.0
  )

  return {
      "shares_held": shares_held,
      "total_invested": round(total_invested, 2),
      "portfolio_value": round(portfolio_value, 2),
      "net_profit": round(net_profit, 2),
      "absolute_return_pct": round(absolute_return_pct, 2),
      "annualized_return_pct": round(annualized_return_pct, 2),
  }


def prepare_indicators(df):
  # 1. Daily MAs
  df["SMA_50_Daily"] = df["Close"].rolling(window=50).mean()
  df["SMA_200_Daily"] = df["Close"].rolling(window=200).mean()

  # 2. Weekly MAs
  df_weekly = df["Close"].resample("W").last().to_frame()
  df_weekly["SMA_50_Weekly"] = df_weekly["Close"].rolling(window=50).mean()
  df_weekly["SMA_200_Weekly"] = df_weekly["Close"].rolling(window=200).mean()

  df = df.join(
      df_weekly[["SMA_50_Weekly", "SMA_200_Weekly"]], how="left"
  ).ffill()

  # Discrete crossunder triggers (Close drops below MA)
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

  return df


def generate_markdown_report(ticker, current_price, results_dict):
  md = f"""# 📊 Moving Average Strategy Comparison Report: `{ticker.upper()}`

> **Testing Rule:** Buy 1 unit every time the price crosses below the specified Moving Average benchmark. Each scenario is tracked independently.

* **Current Market Price:** `${current_price:,.2f}`

---
"""

  scenarios = [
      ("50-Day DMA", "Trigger_50D"),
      ("200-Day DMA", "Trigger_200D"),
      ("50-Week WMA", "Trigger_50W"),
      ("200-Week WMA", "Trigger_200W"),
  ]

  horizons = [1, 5, 10]

  for scen_name, scen_key in scenarios:
    md += f"### Scenario: Buying on Crossunder of **{scen_name}**\n\n"
    md += "| Time Horizon | Shares Held | Invested ($) | Final Value ($) | Absolute Return | Annualized Return |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: |\n"

    for h in horizons:
      res = results_dict.get((scen_key, h))
      if res:
        abs_emoji = "🟢" if res["absolute_return_pct"] >= 0 else "🔴"
        ann_emoji = "🟢" if res["annualized_return_pct"] >= 0 else "🔴"
        md += f"| **{h} Year** | {res['shares_held']} | ${res['total_invested']:,.2f} | ${res['portfolio_value']:,.2f} | {abs_emoji} {res['absolute_return_pct']:+.2f}% | {ann_emoji} {res['annualized_return_pct']:+.2f}% |\n"
      else:
        md += f"| **{h} Year** | N/A | N/A | N/A | N/A | N/A |\n"

    md += "\n"

  md += "*Report automatically generated via GitHub Actions & Python.*"
  return md


if __name__ == "__main__":
  df = fetch_data(TICKER)
  df = prepare_indicators(df)
  current_price = float(df["Close"].iloc[-1])

  scenarios = [
      ("50-Day DMA", "Trigger_50D"),
      ("200-Day DMA", "Trigger_200D"),
      ("50-Week WMA", "Trigger_50W"),
      ("200-Week WMA", "Trigger_200W"),
  ]
  horizons = [1, 5, 10]

  results_dict = {}
  for scen_name, scen_key in scenarios:
    for h in horizons:
      results_dict[(scen_key, h)] = run_scenario_backtest(df, scen_key, h)

  report_md = generate_markdown_report(TICKER, current_price, results_dict)

  # Publish to GitHub Actions Job Summary
  github_summary_path = os.getenv("GITHUB_STEP_SUMMARY")
  if github_summary_path:
    with open(github_summary_path, "a") as summary_file:
      summary_file.write(report_md)
    print("Comparative report successfully published to GitHub Actions Job Summary!")
  else:
    print(report_md)
