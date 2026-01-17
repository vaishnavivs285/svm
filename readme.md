# SVM Text Classification using Web Scraping

## Project Overview
This project demonstrates an **end-to-end machine learning pipeline** where text data is **web scraped**, **cleaned**, **converted into numerical features**, and finally **classified using an SVM (Support Vector Machine)** model.

Instead of using Kaggle datasets, real-world data was collected from the web to understand practical challenges such as noisy text, class imbalance, and weak labeling.
We scraped the data from Hacker News — specifically the front page of news.ycombinator.com.(https://news.ycombinator.com/)

---

## Problem Statement
To build a text classification system that:
- Scrapes news headlines from the web
- Cleans and preprocesses raw text
- Converts text into numerical form using TF-IDF
- Trains an SVM classifier with hyperparameter tuning
- Evaluates performance using proper ML metrics

---

## Project Structure
```
svm_webscrapping/
│
├── data/
│   ├── raw_news.csv          # Scraped raw headlines
│   └── processed_news.csv    # Cleaned text with sentiment labels
│
├── src/
│   ├── scrape.py             # Web scraping logic (BeautifulSoup)
│   ├── preprocess.py         # Text cleaning & labeling
│   └── train.py              # TF-IDF + SVM training & evaluation
│
├── requirements.txt
└── README.md
```

---

## Data Collection (Web Scraping)
- Used **requests** to fetch HTML pages
- Used **BeautifulSoup** to parse and extract headlines
- Scraped multiple static web pages
- Stored headlines in `raw_news.csv`

Why web scraping?
> To simulate real-world data collection instead of relying on pre-cleaned datasets.

---

## Text Preprocessing
Steps applied:
1. Lowercasing text
2. Removing special characters and numbers
3. Removing stopwords (NLTK)
4. Creating a clean text column

### Weak Labeling (Rule-Based)
Since no labeled data was available, a **rule-based sentiment labeling** approach was used:
- Positive keywords → `positive`
- Negative keywords → `negative`
- Otherwise → `neutral`

This generates **weak labels**, which is common in early-stage ML projects.

---

## Feature Engineering (TF-IDF)
Used **TF-IDF Vectorizer** to convert text into numerical vectors.

Why TF-IDF?
- ML models cannot understand text directly
- TF-IDF gives higher weight to informative words
- Works very well with linear models like SVM

---

## Model Training (SVM)
- Model: `LinearSVC`
- Baseline model trained first
- Hyperparameter tuning using **GridSearchCV**
- Evaluation metric: `f1_macro` (handles class imbalance better)

---

## Evaluation Metrics
The following metrics were used:
- **Classification Report** (Precision, Recall, F1-score)
- **Confusion Matrix**
- **Class Distribution Plot**

### Important Observation
Due to:
- Small dataset size
- Heavy class imbalance

The test set contained only the `neutral` class, resulting in a **1×1 confusion matrix**.

This is a **data limitation**, not a modeling or coding issue.

---

## Visualization
- Class distribution bar chart
- Confusion matrix heatmap

These visualizations help understand:
- Dataset imbalance
- Model prediction behavior

---

## Key Challenges Faced
- Scraping dynamic vs static websites
- Very small and imbalanced dataset
- Weak labels instead of ground truth
- Sparse confusion matrix

These challenges reflect **real-world ML problems**.

---

## Learnings
- ML pipelines depend heavily on data quality
- SVM performs well on high-dimensional sparse text data
- Evaluation metrics can be misleading on small datasets
- Class imbalance affects confusion matrices significantly

---

## Future Improvements
- Scrape larger datasets
- Improve sentiment labeling quality
- Use stratified sampling
- Try advanced models (Logistic Regression, Naive Bayes)

---

## Tools & Libraries
- Python
- BeautifulSoup
- Pandas
- NLTK
- Scikit-learn
- Matplotlib

---

## Conclusion
This project showcases a **complete real-world ML workflow**, focusing on understanding data limitations, model behavior, and proper evaluation rather than chasing high accuracy.

---



