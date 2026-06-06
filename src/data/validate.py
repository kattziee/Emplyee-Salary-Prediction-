def validate_data(df):

    if "Salary" not in df.columns:
        raise ValueError(
            "Salary column not found"
        )

    df = df.dropna(
        subset=["Salary"]
    )

    df = df[
        df["Salary"] > 0
    ]

    return df