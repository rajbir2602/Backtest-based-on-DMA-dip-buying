import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import numpy as np
import pandas as pd
import yfinance as yf

# Configuration via Environment Variables (configured in GitHub Secrets)
TICKER = os.getenv("STOCK_TICKER", "AAPL")
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT_EMAIL = os.getenv("RECIPIENT_EMAIL", GMAIL_USER)


def fetch_data(ticker):
  print(f"Fetching data for {ticker}...")
  # Fetch max historical daily data to compute daily and weekly indicators accurately
  df = yf.download(ticker, period="max", interval="1d", progress=False)

  if df.empty:
    raise ValueError(f"No data found for ticker: {ticker}")

  # Flatten columns if multi-index is returned by yfinance
  if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

  df = df[["Close"]].dropna()
  return df


def run_backtest(df):
  # 1. Compute Daily Moving Averages
  df["SMA_50_Daily"] = df["Close"].rolling(window=50).mean()
  df["SMA_200_Daily"] = df["Close"].rolling(window=200).mean()

  # 2. Resample to Weekly to compute Weekly Moving Averages
  df_weekly = df["Close"].resample("W").last().to_frame()
  df_weekly["SMA_50_Weekly"] = df_weekly["Close"].rolling(window=50).mean()
  df_weekly["SMA_200_Weekly"] = df_weekly["Close"].rolling(window=200).mean()

  # Merge weekly indicators back to daily dataframe via forward fill
  df = df.join(
      df_weekly[["SMA_50_Weekly", "SMA_200_Weekly"]], how="left"
  ).ffill()

  # Identify crossing/touching events (Close < MA)
  # To avoid buying every single day the price stays below, we can trigger a buy
  # when the close crosses below the MA, or every day it trades below depending on strictness.
  # Here, we trigger a buy on days where Close < MA AND previous Close >= MA (Discrete Crossunder),
  # or continuously if you prefer accumulation on any day below. Let's use discrete crossunder
  # for realistic accumulation triggers, or daily conditions.
  # User prompt: "Everytime I buy a stock when it goes below its [MAs]... buy one stick qty."

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
  cash_flow_records = []

  for date, row in sub_df.iterrows():
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

  # Annualized Return (CAGR approximation based on holding period cash flows or simple CAGR)
  # Standard CAGR on total invested vs final value can be skewed by periodic accumulation.
  # Using Modified Dietz or XIRR style annualized approximation:
  if total_invested > 0 and years > 0:
    annualized_return_pct = (
        (portfolio_value / total_invested) ** (1 / years) - 1
    ) * 100
  else:
    annualized_return_pct = 0.0

  return {
      "horizon": f"{years} Year" if years > 1 else "1 Year",
      "shares_held": shares_held,
      "total_invested": round(total_invested, 2),
      "portfolio_value": round(portfolio_value, 2),
      "net_profit": round(net_profit, 2),
      "absolute_return_pct": round(absolute_return_pct, 2),
      "annualized_return_pct": round(annualized_return_pct, 2),
  }


def generate_html_report(ticker, current_price, metrics_list):
  html = f"""
    <html>
      <head>
        <style>
          body {{ font-family: Arial, sans-serif; color: #333333; line-height: 1.6; }}
          .container {{ max-width: 600px; margin: 0 auto; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; background-color: #f9f9f9; }}
          h2 {{ color: #1a73e8; border-bottom: 2px solid #1a73e8; padding-bottom: 8px; }}
          .summary-box {{ background: #ffffff; padding: 15px; border-radius: 6px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
          table {{ width: 100%; border-collapse: collapse; background: #ffffff; border-radius: 6px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
          th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #eeeeee; }}
          th {{ background-color: #1a73e8; color: #ffffff; font-weight: 600; }}
          tr:hover {{ background-color: #f1f3f4; }}
          .positive {{ color: #0d652d; font-weight: bold; }}
          .negative {{ color: #c5221f; font-weight: bold; }}
          .footer {{ margin-top: 20px; font-size: 12px; color: #777777; text-align: center; }}
        </style>
      </head>
      <body>
        <div class="container">
          <h2>Moving Average Accumulation Strategy Report</h2>
          <div class="summary-box">
            <p><strong>Target Ticker:</strong> {ticker}</p>
            <p><strong>Current Market Price:</strong> ${current_price:.2f}</p>
            <p><strong>Strategy Rule:</strong> Buy 1 unit whenever the price crosses below the 50-day, 200-day, 50-week, or 200-week moving averages.</p>
          </div>
          
          <table>
            <tr>
              <th>Time Horizon</th>
              <th>Shares Held</th>
              <th>Invested ($)</th>
              <th>Final Value ($)</th>
              <th>Absolute Return</th>
              <th>Annualized Return</th>
            </tr>
  """

  for m in metrics_list:
    abs_class = "positive" if m["absolute_return_pct"] >= 0 else "negative"
    ann_class = "positive" if m["annualized_return_pct"] >= 0 else "negative"
    html += f"""
            <tr>
              <td><strong>{m['horizon']}</strong></td>
              <td>{m['shares_held']}</td>
              <td>${m['total_invested']:,.2f}</td>
              <td>${m['portfolio_value']:,.2f}</td>
              <td class="{abs_class}">{m['absolute_return_pct']:+.2f}%</td>
              <td class="{ann_class}">{m['annualized_return_pct']:+.2f}%</td>
            </tr>
    """

  html += f"""
          </table>
          <div class="footer">
            <p>Automated report generated via GitHub Actions & Python.</p>
          </div>
        </div>
      </body>
    </html>
    """
  return html


def send_email(html_content, ticker):
  msg = MIMEMultipart("alternative")
  msg["Subject"] = (
      f"📈 MA Accumulation Strategy Backtest Report: {ticker.upper()}"
  )
  msg["From"] = GMAIL_USER
  msg["To"] = RECIPIENT_EMAIL

  msg.attach(MIMEText(html_content, "html"))

  print(f"Sending email report to {RECIPIENT_EMAIL}...")
  with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
    server.login(GMAIL_USER, GMAIL_APP_PASSWORD)
    server.sendmail(GMAIL_USER, RECIPIENT_EMAIL, msg.as_string())
  print("Email sent successfully!")


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

  html_report = generate_html_report(TICKER, current_price, metrics_list)
  send_email(html_report, TICKER)
