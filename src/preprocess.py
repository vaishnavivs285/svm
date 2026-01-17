import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
# Step 1: Load scraped data
df = pd.read_csv("data/raw_news.csv")

print(df.head())
# Step 2: Define stopwords
stop_words = set(stopwords.words("english"))
# Step 3: Text cleaning function
def clean_text(text):
    text = text.lower()                      # lowercase
    text = re.sub(r"[^a-z\s]", "", text)     # remove special characters
    words = text.split()                     # split into words
    words = [w for w in words if w not in stop_words]  # remove stopwords
    return " ".join(words)
#step 4 Apply cleaning FIRST
df["clean_text"] = df["headline"].apply(clean_text)
# Step 5: Simple rule-based sentiment labeling
def label_sentiment(text):
    if any(word in text for word in ["growth", "profit", "gain", "rise", "success", "boost"]):
        return "positive"
    elif any(word in text for word in ["crisis", "loss", "fall", "decline", "fail", "risk"]):
        return "negative"
    else:
        return "neutral"
df["sentiment"] = df["clean_text"].apply(label_sentiment)  

print(df[["headline", "clean_text"]].head())
# Step 6: Save processed data
df.to_csv("data/processed_news.csv", index=False)

print("Preprocessing completed. Clean data with labels saved.")

