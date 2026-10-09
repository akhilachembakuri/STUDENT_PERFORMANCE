
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create dataset
data = {
    "study_hours": [1,2,3,4,5,6,7,8,2,4,6,3,5,7,9,10,1,8,9,6],
    "attendance": [50,55,60,65,75,80,90,58,68,78,85,92,95,45,88,94,96,76,82,70],
    "previous_marks": [35,40,45,55,60,65,75,42,52,62,48,58,70,80,85,30,78,82,64,55],
    "final_marks": [30,35,42,48,55,63,68,85,38,50,65,44,58,75,88,95,25,82,90,66]
}

# Create DataFrame
df = pd.DataFrame(data)

# Select input features and target
X = df[["study_hours", "attendance", "previous_marks"]]
y = df["final_marks"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Evaluate model
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save trained model
joblib.dump(model, "students_model.pkl")

print("Model saved as students_model.pkl")
print("Saved successfully!")