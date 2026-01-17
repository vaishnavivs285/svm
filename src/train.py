import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt 
# Load preprocessed data
df = pd.read_csv("data/processed_news.csv")
print(df[["clean_text", "sentiment"]].head())
# class distribution
df["sentiment"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=3000)

# Convert text to numerical features
X = vectorizer.fit_transform(df["clean_text"])
y = df["sentiment"]
# Split data into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.svm import LinearSVC
# Create SVM model
svm = LinearSVC(C=1.0,class_weight="balanced")

# Train model
svm.fit(X_train, y_train)
#Evaluate basic model 
from sklearn.metrics import classification_report
# Predict on test data
y_pred = svm.predict(X_test)

print(classification_report(y_test, y_pred))
#hyperparameter tuning
from sklearn.model_selection import GridSearchCV
# Hyperparameter grid
param_grid = {
    "C": [0.01, 0.1, 1, 10]
}
grid = GridSearchCV(
    LinearSVC(),
    param_grid,
    cv=5,
    scoring="f1_macro"
)
#train gridsearch 
grid.fit(X_train, y_train)
#get best model
best_model = grid.best_estimator_
print("Best C:", grid.best_params_)
#Top TF-IDF Words per Class
import numpy as np
feature_names = vectorizer.get_feature_names_out()
coef = best_model.coef_
classes = best_model.classes_
top_n = 10
for i, class_label in enumerate(classes):
    print(f"\nTop words for class: {class_label}")
    top_indices = np.argsort(coef[i])[-top_n:]
    for idx in top_indices:
        print(feature_names[idx])

#final evaluation
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))
#confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, y_pred))
#visulization
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    best_model,
    X_test,
    y_test,
    cmap="Blues"
)
plt.title("Confusion Matrix - SVM")
plt.show()

