import requests
from bs4 import BeautifulSoup
import pandas as pd

url = "https://news.ycombinator.com/"
response = requests.get(url)

print("Status code:", response.status_code)

soup = BeautifulSoup(response.text, "html.parser")

# Hacker News headlines
titles = soup.find_all("span", class_="titleline")

news = []

for t in titles:
    text = t.get_text(strip=True)
    if text:
        news.append(text)

print("Total headlines scraped:", len(news))
print("Sample headlines:", news[:5])

df = pd.DataFrame(news, columns=["headline"])
df.to_csv("data/raw_news.csv", index=False)

print("Saved rows:", len(df))

