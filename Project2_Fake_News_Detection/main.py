import pandas as pd
import pickle
import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("dataset.csv")

# ==========================
# EDA SECTION
# ==========================

print("\n========== DATASET INFORMATION ==========")
data.info()

print("\n========== MISSING VALUES ==========")
print(data.isnull().sum())

# Remove missing values
data = data.dropna()

print("\n========== DATASET STATISTICS ==========")
print(data.describe(include='all'))

print("\n========== LABEL DISTRIBUTION ==========")
print(data["label"].value_counts())

# Text length analysis
data["text_length"] = data["text"].apply(len)

print("\nAverage News Length:")
print(round(data["text_length"].mean(), 2))

# ==========================
# TEXT PREPROCESSING
# ==========================

def preprocess_text(text):
    text = text.lower()

    text = "".join(
        char for char in text
        if char not in string.punctuation
    )

    words = text.split()

    stop_words = set(stopwords.words('english'))

    words = [
        word for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# Apply preprocessing
data["clean_text"] = data["text"].apply(preprocess_text)

# ==========================
# FEATURE SELECTION
# ==========================

X = data["clean_text"]
y = data["label"]

# ==========================
# TF-IDF VECTORIZATION
# ==========================

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

print("\n========== SAMPLE TF-IDF WORDS ==========")
print(vectorizer.get_feature_names_out()[:20])

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# MODEL TRAINING
# ==========================

model = LogisticRegression()

model.fit(X_train, y_train)

# ==========================
# MODEL EVALUATION
# ==========================

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL RESULTS ==========")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, predictions))

# ==========================
# MODEL STORAGE
# ==========================

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("\nModel saved successfully!")
print("Files created:")
print("1. model.pkl")
print("2. vectorizer.pkl")

# ==========================
# VISUALIZATION
# ==========================

data["label"].value_counts().plot(kind="bar")

plt.title("Fake vs Real News Distribution")
plt.xlabel("News Type")
plt.ylabel("Count")

plt.show()

# ==========================
# USER INPUT PREDICTION
# ==========================

print("\n========== NEWS CLASSIFICATION ==========")

news = input("\nEnter News Text:\n")

news_clean = preprocess_text(news)

news_vector = vectorizer.transform([news_clean])

prediction = model.predict(news_vector)[0]

confidence = max(
    model.predict_proba(news_vector)[0]
) * 100

print("\n==============================")
print("PREDICTION RESULT")
print("==============================")
print("News Type:", prediction)
print("Confidence Score:", round(confidence, 2), "%")
print("==============================")

# ==========================
# TEST MODEL LOADING
# ==========================

loaded_model = pickle.load(open("model.pkl", "rb"))
loaded_vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

print("\nSaved model loaded successfully!")
