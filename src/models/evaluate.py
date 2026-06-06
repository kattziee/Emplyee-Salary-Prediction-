import joblib

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.model_selection import (
    train_test_split
)

from src.config.config import *

from src.data.ingest import (
    load_data
)

df = load_data(DATA_PATH)

X = df.drop(
    "Salary",
    axis=1
)

y = df["Salary"]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
)

model = joblib.load(
    MODEL_PATH
)

predictions = model.predict(
    X_test
)

print(
    "MAE:",
    mean_absolute_error(
        y_test,
        predictions
    )
)

print(
    "R2:",
    r2_score(
        y_test,
        predictions
    )
)