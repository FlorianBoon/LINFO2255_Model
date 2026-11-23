import json
from pathlib import Path
import pandas as pd
import pyreadr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib


def analyze_data(df):
    """
    Analyze the DataFrame and print basic information.
    """
    print(df.head())
    print(df.shape)
    print(df.dtypes)
    print(df.isnull().sum())
    

def plot_attendance_by_year(df):
    """Plot attendance for the last three years and COVID years."""

    # Keep valid observations
    df = df[df["n_expected"] > 0].copy()

    # Aggregate attendance by year and week
    def aggregate_attendance(data):
        weekly = (
            data.groupby(
                ["academic_year", "week_since_start"],
                observed=True
            )
            .agg(
                attended=("n_attended", "sum"),
                expected=("n_expected", "sum"),
            )
            .reset_index()
        )

        weekly["attendance_rate"] = (
            weekly["attended"] / weekly["expected"]
        )

        return weekly

    # Last three academic years
    last_three_years = sorted(
        df["academic_year"].dropna().unique()
    )[-3:]

    last_three_years_data = aggregate_attendance(
        df[df["academic_year"].isin(last_three_years)]
    )

    # COVID observations only
    covid_df = df[
        df["covid_period"] == "During COVID"
    ]

    covid_data = aggregate_attendance(covid_df)

    # Plot both graphs
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 6), sharey=True
    )

    datasets = [
        (last_three_years_data, "Last three academic years"),
        (covid_data, "COVID period"),
    ]

    for ax, (data, title) in zip(axes, datasets):
        for year, year_data in data.groupby("academic_year"):
            ax.plot(
                year_data["week_since_start"],
                year_data["attendance_rate"],
                marker="o",
                label=str(int(year)),
            )

        ax.set_xlabel("Week since start")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(title="Academic year")

    axes[0].set_ylabel("Attendance rate")
    fig.suptitle("Attendance by Week and Academic Year")

    plt.tight_layout()
    plt.show()



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
    analyze_data(df)
    plot_attendance_by_year(df)
    model = train_model(df)