import pandas as pd
import requests
import yfinance as yf
from io import StringIO
from pandas.tseries.offsets import BDay


# ------------------------------------------------------------
# OpenInsider sales screener URL
# ------------------------------------------------------------

url = (
    "http://www.openinsider.com/screener?"
    "s=&o=&pl=&ph=&ll=&lh=100&fd=-1"
    "&fdr=05%2F01%2F2021+-+05%2F09%2F2025"
    "&td=0&tdr=&fdlyl=&fdlyh=&daysago="
    "&xs=1"
    "&vl=100&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999"
    "&isofficer=1&iscob=1&isceo=1&ispres=1&iscoo=1&iscfo=1&isgc=1&isvp=1"
    "&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h="
    "&sortcol=0&cnt=500&page=1"
)

headers = {
    "User-Agent": "Mozilla/5.0 research script"
}


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def parse_money(x):
    """
    Parse values like '$100,000', '+$100,000', '($100,000)'.
    For sales, return absolute dollar value sold.
    """
    if pd.isna(x):
        return None

    s = str(x)
    s = s.replace("$", "")
    s = s.replace(",", "")
    s = s.replace("+", "")
    s = s.replace("(", "-").replace(")", "")
    s = s.strip()

    if s in {"", "-"}:
        return None

    try:
        return abs(float(s))
    except ValueError:
        return None


def parse_qty(x):
    if pd.isna(x):
        return None

    s = str(x)
    s = s.replace(",", "")
    s = s.replace("+", "")
    s = s.replace("(", "-").replace(")", "")
    s = s.strip()

    if s in {"", "-"}:
        return None

    try:
        return abs(float(s))
    except ValueError:
        return None


def clean_columns(table):
    """
    OpenInsider uses non-breaking spaces in some headers.
    This converts them to normal spaces.
    """
    table = table.copy()
    table.columns = [
        str(c).replace("\xa0", " ").strip()
        for c in table.columns
    ]
    return table


def find_openinsider_results_table(tables):
    """
    Find the table that contains the actual OpenInsider screener results.
    """
    for table in tables:
        table = clean_columns(table)
        cols = list(table.columns)

        if (
            "Filing Date" in cols
            and "Trade Date" in cols
            and "Ticker" in cols
            and "Company Name" in cols
            and "Trade Type" in cols
            and "Value" in cols
        ):
            return table

    raise ValueError("Could not find the OpenInsider results table.")


def empty_return_series(entry_date):
    return pd.Series({
        "entry_date": entry_date.date(),
        "entry_price": None,
        "ret_1w": None,
        "ret_1m": None,
        "ret_3m": None,
        "ret_6m": None,
    })


def get_forward_returns(row):
    ticker = row["ticker"]
    filing_date = pd.to_datetime(row["filing_date"])

    # Conservative rule: enter next business day after public filing.
    # This is not perfect because BDay does not know exchange holidays,
    # but it is fine for a first-pass prototype.
    entry_date = filing_date.normalize() + BDay(1)

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

        if prices.empty:
            return empty_return_series(entry_date)

        # yfinance sometimes returns MultiIndex columns even for one ticker.
        if isinstance(prices.columns, pd.MultiIndex):
            prices.columns = prices.columns.get_level_values(0)

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
        return empty_return_series(entry_date)


# ------------------------------------------------------------
# Download and parse OpenInsider
# ------------------------------------------------------------

print("Downloading OpenInsider results...")

html = requests.get(url, headers=headers, timeout=30).text
tables = pd.read_html(StringIO(html))

df = find_openinsider_results_table(tables)

print("Selected columns:")
print(df.columns.tolist())


# ------------------------------------------------------------
# Rename and clean columns
# ------------------------------------------------------------

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

required = [
    "filing_date", "trade_date", "ticker", "company",
    "insider", "title", "trade_type", "price", "qty", "value"
]

missing = [c for c in required if c not in df.columns]

if missing:
    print("Missing columns:", missing)
    print("Actual columns:", df.columns.tolist())
    raise SystemExit


df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
df["price"] = df["price"].apply(parse_money)
df["qty"] = df["qty"].apply(parse_qty)
df["value"] = df["value"].apply(parse_money)


# ------------------------------------------------------------
# Filter to insider sales
# ------------------------------------------------------------

signals = df[
    df["trade_type"].astype(str).str.contains("S - Sale", na=False)
    & (df["value"] >= 100_000)
].copy()

# Optional: uncomment this if you want CEO-only sales.
# signals = signals[
#     signals["title"].astype(str).str.contains("CEO", case=False, na=False)
# ].copy()

signals = signals.sort_values("filing_date", ascending=False)

print("\nSales signals:")
print(signals[[
    "filing_date", "trade_date", "ticker", "company", "insider", "title",
    "trade_type", "price", "qty", "value"
]].head(25))

signals.to_csv("openinsider_officer_sales.csv", index=False)
print(f"\nSaved {len(signals)} signals to openinsider_officer_sales.csv")


# ------------------------------------------------------------
# Fetch prices and calculate forward returns
# ------------------------------------------------------------

print("\nFetching forward returns...")

returns = signals.apply(get_forward_returns, axis=1)

results = pd.concat(
    [signals.reset_index(drop=True), returns.reset_index(drop=True)],
    axis=1
)

print("\nResults:")
print(results[[
    "filing_date", "entry_date", "ticker", "company", "insider", "title",
    "value", "entry_price", "ret_1w", "ret_1m", "ret_3m", "ret_6m"
]].head(25))

results.to_csv("openinsider_officer_sales_with_returns.csv", index=False)
print(f"\nSaved {len(results)} rows to openinsider_officer_sales_with_returns.csv")


# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

summary_cols = ["ret_1w", "ret_1m", "ret_3m", "ret_6m"]

print("\nSummary:")
for col in summary_cols:
    s = results[col].dropna()

    print(f"{col}:")
    print(f"  count:  {len(s)}")

    if len(s) == 0:
        print("  mean:   n/a")
        print("  median: n/a")
        print("  win %:  n/a")
        continue

    print(f"  mean:   {s.mean():.2%}")
    print(f"  median: {s.median():.2%}")
    print(f"  win %:  {(s > 0).mean():.2%}")


# ------------------------------------------------------------
# Optional diagnostics
# ------------------------------------------------------------

print("\nTop 10 positive 1-week returns after sales:")
print(results.sort_values("ret_1w", ascending=False)[[
    "filing_date", "entry_date", "ticker", "company", "insider", "title",
    "value", "ret_1w"
]].head(10))

print("\nTop 10 negative 1-week returns after sales:")
print(results.sort_values("ret_1w", ascending=True)[[
    "filing_date", "entry_date", "ticker", "company", "insider", "title",
    "value", "ret_1w"
]].head(10))
