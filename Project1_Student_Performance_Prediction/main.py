import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("dataset.csv")

X = data[["attendance", "marks", "study_hours"]]
y = data["final_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = r2_score(y_test, predictions)

print("Student Performance Prediction System")
print("-" * 40)
print("Model Accuracy:", round(accuracy, 2))

attendance = float(input("Enter Attendance (%): "))
marks = float(input("Enter Marks: "))
study_hours = float(input("Enter Study Hours per Day: "))

result = model.predict([[attendance, marks, study_hours]])

print("\nPredicted Final Score:", round(result[0], 2))
