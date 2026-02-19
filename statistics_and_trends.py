

from corner import corner  # imported in template; not required to use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns


def plot_relational_plot(df):
    """
    Relational plot: shows relationship/trend over time.
    Here: line plot of cumulative_total_cases over date for a chosen country.
    """
    # Choose a country with substantial data (fallback to the most common country)
    preferred = "United States"
    if "country" in df.columns and preferred in df["country"].unique():
        country_name = preferred
    else:
        country_name = df["country"].value_counts().index[0]

    # Prepare series for that country
    sub = df[df["country"] == country_name].copy()
    sub = sub.sort_values("date")

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=sub,
        x="date",
        y="cumulative_total_cases",
        ax=ax
    )
    ax.set_title(f"Cumulative Total Cases Over Time ({country_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Total Cases")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("relational_plot.png")
    plt.close(fig)
    return


def plot_categorical_plot(df):
    """
    Categorical plot: compares categories.
    Here: bar chart of Top 10 countries by cumulative_total_cases (latest date per country).
    """
    # Latest record per country (max date)
    latest = (
        df.sort_values("date")
          .groupby("country", as_index=False)
          .tail(1)
          .copy()
    )

    top10 = latest.nlargest(10, "cumulative_total_cases")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=top10,
        x="cumulative_total_cases",
        y="country",
        ax=ax
    )
    ax.set_title("Top 10 Countries by Cumulative Total Cases (Latest Date)")
    ax.set_xlabel("Cumulative Total Cases")
    ax.set_ylabel("Country")
    plt.tight_layout()
    plt.savefig("categorical_plot.png")
    plt.close(fig)
    return


def plot_statistical_plot(df):
    """
    Statistical plot: shows statistical relationships/distribution properties.
    Here: correlation heatmap across numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number]).copy()
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    plt.savefig("statistical_plot.png")
    plt.close(fig)
    return


def statistical_analysis(df, col: str):
    """
    Computes the first four statistical moments-related measures:
    - Mean
    - Standard Deviation
    - Skewness
    - Excess Kurtosis
    """
    series = df[col].dropna()

    mean = series.mean()
    stddev = series.std(ddof=1)

    # scipy expects array-like; cast to numpy
    skew = ss.skew(series.to_numpy(), bias=False)
    excess_kurtosis = ss.kurtosis(series.to_numpy(), fisher=True, bias=False)

    return mean, stddev, skew, excess_kurtosis


def preprocessing(df):
    """
    Preprocess/clean the dataset:
    - Parse date
    - Drop duplicates
    - Fill missing daily values with 0
    - Ensure numeric columns are numeric
    - Remove impossible negative values (clip at 0)
    - Print quick exploratory outputs (head/describe/corr)
    """
    # Quick looks (as requested in template)
    print("Head:\n", df.head())
    print("\nDescribe (numeric):\n", df.describe(include=[np.number]))

    # Parse date safely
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows with invalid dates (cannot plot time series otherwise)
    if "date" in df.columns:
        df = df.dropna(subset=["date"])

    # Drop duplicate rows
    df = df.drop_duplicates()

    # Ensure numeric columns are numeric
    num_cols = [
        "cumulative_total_cases",
        "daily_new_cases",
        "active_cases",
        "cumulative_total_deaths",
        "daily_new_deaths",
    ]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Fill missing daily values with 0 (common in COVID reporting)
    for c in ["daily_new_cases", "daily_new_deaths"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Fill missing cumulative/active with 0 (safer than dropping huge rows)
    for c in ["cumulative_total_cases", "active_cases", "cumulative_total_deaths"]:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # Clip negatives (some datasets contain corrections; for this assignment we keep non-negative)
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    # Correlation quick feature
    print("\nCorrelation (numeric):\n", df.corr(numeric_only=True))

    return df


def writing(moments, col):
    print(f"For the attribute {col}:")
    print(
        f"Mean = {moments[0]:.2f}, "
        f"Standard Deviation = {moments[1]:.2f}, "
        f"Skewness = {moments[2]:.2f}, and "
        f"Excess Kurtosis = {moments[3]:.2f}."
    )

    skew_val = moments[2]
    kurt_val = moments[3]

    # Skewness interpretation
    # (keep simple + readable; thresholds can vary in literature)
    if skew_val > 0.5:
        skew_text = "right skewed"
    elif skew_val < -0.5:
        skew_text = "left skewed"
    else:
        skew_text = "not skewed"

    # Excess kurtosis interpretation (Fisher: 0 ~ normal/mesokurtic)
    if kurt_val > 0.5:
        kurt_text = "leptokurtic"
    elif kurt_val < -0.5:
        kurt_text = "platykurtic"
    else:
        kurt_text = "mesokurtic"

    print(f"The data was {skew_text} and {kurt_text}.")
    return


def main():
    df = pd.read_csv("data.csv")
    df = preprocessing(df)

    # Choose a numeric column for the 4-moment analysis
    # This dataset supports: daily_new_cases, daily_new_deaths, active_cases, etc.
    col = "daily_new_cases"

    plot_relational_plot(df)
    plot_statistical_plot(df)
    plot_categorical_plot(df)

    moments = statistical_analysis(df, col)
    writing(moments, col)
    return


if __name__ == "__main__":
    main()
