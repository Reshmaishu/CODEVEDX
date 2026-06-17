import pandas as pd
import pickle
import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

nltk.download('stopwords')


data = pd.read_csv("dataset.csv")

print("\n========== DATASET INFORMATION ==========")
data.info()

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

data.dropna(inplace=True)

print("\n========== LABEL DISTRIBUTION ==========")
print(data["label"].value_counts())


data["text_length"] = data["text"].apply(len)

print("\nAverage News Length:")
print(round(data["text_length"].mean(), 2))

plt.figure(figsize=(6,4))
data["label"].value_counts().plot(kind="bar")
plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.show()


stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = str(text).lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

data["clean_text"] = data["text"].apply(preprocess_text)



X = data["clean_text"]
y = data["label"]


vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

print("\n========== SAMPLE TF-IDF WORDS ==========")
print(vectorizer.get_feature_names_out()[:20])



X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)




model = LogisticRegression(
    max_iter=1000
)

model.fit(X_train, y_train)


predictions = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n========== MODEL RESULTS ==========")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions
))

print("\nConfusion Matrix:")
print(confusion_matrix(
    y_test,
    predictions
))



pickle.dump(
    model,
    open("model.pkl", "wb")
)

pickle.dump(
    vectorizer,
    open("vectorizer.pkl", "wb")
)

print("\nModel saved successfully!")
print("Files created:")
print("1. model.pkl")
print("2. vectorizer.pkl")



print("\n========== FAKE NEWS DETECTOR ==========")

while True:

    news = input(
        "\nEnter News Text (or type EXIT):\n"
    )

    if news.upper() == "EXIT":
        break

    news_clean = preprocess_text(news)

    news_vector = vectorizer.transform(
        [news_clean]
    )

    prediction = model.predict(
        news_vector
    )[0]

    confidence = max(
        model.predict_proba(
            news_vector
        )[0]
    ) * 100

    print("\n==============================")
    print("PREDICTION RESULT")
    print("==============================")
    print("News Type :", prediction)
    print(
        "Confidence:",
        round(confidence, 2),
        "%"
    )
    print("==============================")


loaded_model = pickle.load(
    open("model.pkl", "rb")
)

loaded_vectorizer = pickle.load(
    open("vectorizer.pkl", "rb")
)

print(
    "\nSaved model loaded successfully!"
)
