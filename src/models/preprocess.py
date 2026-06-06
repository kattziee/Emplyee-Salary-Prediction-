import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

def load_data(path):
    return pd.read_csv(path)

def create_preprocessor(df):

    X = df.drop("Salary", axis=1)

    numeric_features = [
        "Age",
        "Years of Experience"
    ]

    categorical_features = [
        "Gender",
        "Education Level",
        "Job Title"
    ]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer",
             SimpleImputer(strategy="median"))
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer",
             SimpleImputer(strategy="most_frequent")),

            ("encoder",
             OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    return preprocessor