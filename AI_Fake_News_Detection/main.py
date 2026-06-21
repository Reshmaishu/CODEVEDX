import pandas as pd
import pickle
import nltk
import string
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("dataset.csv")

# Text preprocessing function
def clean_text(text):
    text = text.lower()

    # Remove punctuation
    for p in string.punctuation:
        text = text.replace(p, "")

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()

    filtered_words = []

    for word in words:
        if word not in stop_words:
            filtered_words.append(word)

    return " ".join(filtered_words)

# Apply preprocessing
data["clean_text"] = data["text"].apply(clean_text)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data["clean_text"])

y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n================================")
print("AI FAKE NEWS DETECTION TOOL")
print("================================")
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Save model and vectorizer
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

print("Model saved successfully!")
print("Vectorizer saved successfully!")

# Accuracy chart
plt.figure(figsize=(5, 4))
plt.bar(["Accuracy"], [accuracy * 100])
plt.ylabel("Percentage")
plt.title("Model Accuracy")
plt.show()

# User prediction
print("\n===== NEWS PREDICTION =====")

news = input("Enter News Text: ")

clean_news = clean_text(news)

news_vector = vectorizer.transform([clean_news])

prediction = model.predict(news_vector)[0]

confidence = max(model.predict_proba(news_vector)[0]) * 100

print("\nPrediction:", prediction)
print("Confidence Score:", round(confidence, 2), "%")
