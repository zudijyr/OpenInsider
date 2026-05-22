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

# Usually the main results table is one of the last/larger tables.
for i, table in enumerate(tables):
    print(i, table.shape, table.columns.tolist()[:5])

df = max(tables, key=len)

print(df.head())