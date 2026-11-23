import json
from pathlib import Path
import pandas as pd
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor



def analyze_data(df):
    """
    Analyze the DataFrame and print basic information.
    """
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())



def clean_model(df):
    df = df.copy()

    # Convert week to datetime
    df["week"] = pd.to_datetime(df["week"])

    groups = [
        "course_uid",
        "academic_year",
        "academic_quarter",
    ]

    first_week = df.groupby(
        groups, observed=True
    )["week"].transform("min")

    df["week_since_start"] = (
        (df["week"] - first_week).dt.days // 7 + 1
    )

    df = df[df["n_expected"] > 0].copy()

    df["attendance_rate"] = (
        df["n_attended"] / df["n_expected"]
    )
    # Sort observations chronologically
    df = df.sort_values(groups + ["week"])

    grouped = df.groupby(groups, observed=True)

    # Previous attendance rate
    df["previous_attendance_rate"] = (
        grouped["attendance_rate"].shift(1)
    )

    # Previous recorded week
    df["previous_week"] = (
        grouped["week"].shift(1)
    )

    # Time elapsed since previous observation
    df["weeks_since_last_record"] = (
        (df["week"] - df["previous_week"]).dt.days / 7
    )

    return df



def train_model(df):
    features = [
        "week_since_start",
        "previous_attendance_rate",
        "weeks_since_last_record",
        "domain",
        "academic_quarter",
    ]

    data = df[
        (df["covid_period"] == "Post-COVID")
        & (df["previous_attendance_rate"].notna())
    ].copy()

    train = data[data["academic_year"] < 2024]
    test = data[data["academic_year"] == 2024]

    X_train = pd.get_dummies(train[features], dtype=float)
    X_test = pd.get_dummies(test[features], dtype=float)

    X_test = X_test.reindex(
        columns=X_train.columns, fill_value=0
    )

    y_train = train["attendance_rate"]
    y_test = test["attendance_rate"]

    model = HistGradientBoostingRegressor(
        max_iter=100,
        max_leaf_nodes=15,
        learning_rate=0.05,
        random_state=42,
    )    
    model.fit(X_train, y_train)

    predictions = model.predict(X_test).clip(0, 1)

    print("Ridge MAE:", mean_absolute_error(y_test, predictions))
    print("Baseline MAE:", mean_absolute_error(
        y_test, test["previous_attendance_rate"]
    ))

    return model

if __name__ == "__main__":
    df = pyreadr.read_r("data/agg_data_weekly(1).rds")[None]
    df = clean_model(df)
    model = train_model(df)