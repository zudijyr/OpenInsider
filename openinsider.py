import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO

url = "http://www.openinsider.com/screener?s=&o=&pl=&ph=&ll=&lh=100&fd=-1&fdr=05%2F01%2F2021+-+05%2F09%2F2025&td=0&tdr=&fdlyl=&fdlyh=&daysago=&xp=1&vl=100&vh=&ocl=&och=&sic1=-1&sicl=100&sich=9999&isofficer=1&iscob=1&isceo=1&ispres=1&iscoo=1&iscfo=1&isgc=1&isvp=1&grp=0&nfl=&nfh=&nil=&nih=&nol=&noh=&v2l=&v2h=&oc2l=&oc2h=&sortcol=0&cnt=100&page=1"

headers = {
    "User-Agent": "Mozilla/5.0 research script; contact: your-email@example.com"
}

html = requests.get(url, headers=headers, timeout=30).text

tables = pd.read_html(StringIO(html))
df = max(tables, key=len)

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

print(signals[[
    "filing_date", "trade_date", "ticker", "insider", "title",
    "price", "qty", "value"
]].head(25))
