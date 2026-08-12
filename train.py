import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 1. Load dataset
df = pd.read_csv(r"C:\Users\Pragya Saini\OneDrive\Desktop\ford-car-price-prediction\ford.csv")

# 2. Select features and target
X = df[
    [
        "model",
        "year",
        "transmission",
        "mileage",
        "fuelType",
        "tax",
        "mpg",
        "engineSize"
    ]
]

y = df["price"]

# 3. Separate categorical and numerical columns
categorical_columns = [
    "model",
    "transmission",
    "fuelType"
]

numerical_columns = [
    "year",
    "mileage",
    "tax",
    "mpg",
    "engineSize"
]

# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)

# 5. Create model pipeline
model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ]
)

# 6. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 7. Train model
model.fit(X_train, y_train)

# 8. Test model
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
# 8. Make predictions
y_pred = model.predict(X_test)

# 9. Calculate evaluation metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nModel Evaluation")
print("-------------------------")
print(f"R2 Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")
print("-------------------------")

# 10. Save trained model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")

print("Model trained successfully!")
print("--------------------------------")
print("R2 Score:", r2)
print("MAE:", mae)
print("RMSE:", rmse)
print("--------------------------------")

joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")

# 9. Save trained model
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")