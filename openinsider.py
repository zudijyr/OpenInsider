import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

import yfinance as yf
from pandas.tseries.offsets import BDay

url = "http://www.openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=100&fd=-1&fdr=05%2F01%2F2021+-+05%2F09%2F2025&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=100&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&isofficer=1&iscob=1&isceo=1&ispres=1&iscoo=1&iscfo=1&isgc=1&isvp=1&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=5000&page=1"

headers = {
    "User-Agent": "Mozilla/5.0 research script; contact: your-email@example.com"
}

html = requests.get(url, headers=headers, timeout=30).text

tables = pd.read_html(StringIO(html))
df = max(tables, key=len)

def get_forward_returns(row):
    ticker = row["ticker"]
    filing_date = pd.to_datetime(row["filing_date"])

    # Conservative rule: enter next business day after public filing
    entry_date = filing_date.normalize() + BDay(1)

    # Pull enough price data for 6 months forward
    start = entry_date.strftime("%Y-%m-%d")
    end = (entry_date + BDay(140)).strftime("%Y-%m-%d")

    try:
        prices = yf.download(
            ticker,
            start=start,
            end=end,
            progress=False,
            auto_adjust=True
        )
        # yfinance sometimes returns MultiIndex columns even for one ticker.
        # Flatten them so prices["Open"].iloc[0] is a scalar.
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

        if prices.empty:
            return pd.Series({
                "entry_date": entry_date.date(),
                "entry_price": None,
                "ret_1w": None,
                "ret_1m": None,
                "ret_3m": None,
                "ret_6m": None,
            })

        entry_price = float(prices["Open"].iloc[0])

        def ret_after_n_days(n):
            if len(prices) <= n:
                return None
            exit_price = float(prices["Close"].iloc[n])
            return (exit_price / entry_price) - 1

        return pd.Series({
            "entry_date": entry_date.date(),
            "entry_price": entry_price,
            "ret_1w": ret_after_n_days(5),
            "ret_1m": ret_after_n_days(21),
            "ret_3m": ret_after_n_days(63),
            "ret_6m": ret_after_n_days(126),
        })

    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return pd.Series({
            "entry_date": entry_date.date(),
            "entry_price": None,
            "ret_1w": None,
            "ret_1m": None,
            "ret_3m": None,
            "ret_6m": None,
        })

def parse_money(x):
    if pd.isna(x):
        return None
    s = str(x).replace("$", "").replace(",", "").replace("+", "").strip()
    if s in {"", "-"}:
        return None
    return float(s)

def parse_qty(x):
    if pd.isna(x):
        return None
    s = str(x).replace(",", "").replace("+", "").strip()
    if s in {"", "-"}:
        return None
    return float(s)

# Clean column names: OpenInsider uses non-breaking spaces
df.columns = [
    str(c)
    .replace("\xa0", " ")
    .strip()
    for c in df.columns
]

print("Cleaned columns:")
print(df.columns.tolist())

df = df.rename(columns={
    "Filing Date": "filing_date",
    "Trade Date": "trade_date",
    "Ticker": "ticker",
    "Company Name": "company",
    "Insider Name": "insider",
    "Title": "title",
    "Trade Type": "trade_type",
    "Price": "price",
    "Qty": "qty",
    "Value": "value",
})


df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
df["price"] = df["price"].apply(parse_money)
df["qty"] = df["qty"].apply(parse_qty)
df["value"] = df["value"].apply(parse_money)

signals = df[
    df["trade_type"].astype(str).str.contains("P - Purchase", na=False)
    & df["title"].astype(str).str.contains("CEO", case=False, na=False)
    & (df["value"] >= 100_000)
].copy()

signals = signals.sort_values("filing_date", ascending=False)
returns = signals.apply(get_forward_returns, axis=1)
results = pd.concat([signals.reset_index(drop=True), returns.reset_index(drop=True)], axis=1)

print(results[[
    "filing_date", "entry_date", "ticker", "insider", "title",
    "value", "entry_price", "ret_1w", "ret_1m", "ret_3m", "ret_6m"
]].head(25))

results.to_csv("openinsider_ceo_purchases_with_returns.csv", index=False)
print(f"Saved {len(results)} rows to openinsider_ceo_purchases_with_returns.csv")

summary_cols = ["ret_1w", "ret_1m", "ret_3m", "ret_6m"]

print("\nSummary:")
for col in summary_cols:
    s = results[col].dropna()
    print(f"{col}:")
    print(f"  count:  {len(s)}")
    print(f"  mean:   {s.mean():.2%}")
    print(f"  median: {s.median():.2%}")
    print(f"  win %:  {(s > 0).mean():.2%}")

print(results.sort_values("ret_1w", ascending=False)[[
    "filing_date", "entry_date", "ticker", "insider", "title", "value", "ret_1w"
]].head(25))

s = results["ret_1w"].dropna().sort_values(ascending=False)

print("Top 5 avg:", s.head(5).mean())
print("Top 10 avg:", s.head(10).mean())
print("All avg:", s.mean())
print("Avg without top 10:", s.iloc[10:].mean())
print("Median:", s.median())
