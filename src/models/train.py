import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    r2_score
)

from xgboost import XGBRegressor

from .preprocess import (
    load_data,
    create_preprocessor
)

df = load_data("data/Salary_Data.csv")

df = df.dropna(subset=["Salary"])

X = df.drop("Salary", axis=1)
y = df["Salary"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

preprocessor = create_preprocessor(df)

model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    random_state=42
)

pipeline = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

mae = mean_absolute_error(
    y_test,
    predictions
)

r2 = r2_score(
    y_test,
    predictions
)

print(f"MAE: {mae}")
print(f"R2 Score: {r2}")

joblib.dump(
    pipeline,
    "models/salary_model.pkl"
)

print("Model Saved")