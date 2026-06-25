import argparse
import logging
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def load_data(path: str, **read_kwargs) -> pd.DataFrame:
    """Load dataset from `path` and return a DataFrame."""
    try:
        df = pd.read_csv(path, **read_kwargs)
        logging.info(
            "Loaded data from %s (rows=%d, cols=%d)", path, df.shape[0], df.shape[1]
        )
        return df
    except FileNotFoundError:
        logging.exception("Input file not found: %s", path)
        raise


def validate_columns(df: pd.DataFrame, required: list) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame, outlier_method: str = "remove") -> pd.DataFrame:
    """Fill missing values and handle outliers for Age.

    outlier_method: one of 'remove', 'clip', 'flag'
    """
    df = df.copy()

    # Fill missing values
    if "Age" in df.columns:
        if not df["Age"].dropna().empty:
            median_age = df["Age"].median()
            df["Age"] = df["Age"].fillna(median_age)
            logging.debug("Filled Age NA with median=%.2f", median_age)
        else:
            logging.debug("Age column empty or all NA; skipping median fill")

    if "Embarked" in df.columns:
        embarked_non_null = df["Embarked"].dropna()
        if not embarked_non_null.empty:
            mode_emb = embarked_non_null.mode()
            if not mode_emb.empty:
                mode_val = mode_emb.iloc[0]
                df["Embarked"] = df["Embarked"].fillna(mode_val)
                logging.debug("Filled Embarked NA with mode=%s", mode_val)
        else:
            logging.debug("Embarked column empty or all NA; skipping mode fill")

    # Outlier handling using IQR on Age
    if "Age" in df.columns:
        age_non_null = df["Age"].dropna()
        if age_non_null.empty:
            logging.debug("Age column empty or all NA; skipping outlier handling")
        else:
            q1 = age_non_null.quantile(0.25)
            q3 = age_non_null.quantile(0.75)
            iqr = q3 - q1
            minimum_age = q1 - (1.5 * iqr)
            maximum_age = q3 + (1.5 * iqr)
            logging.debug("Age IQR bounds: [%.2f, %.2f]", minimum_age, maximum_age)

            if outlier_method == "remove":
                before = len(df)
                df = df[(df["Age"] >= minimum_age) & (df["Age"] <= maximum_age)]
                logging.info("Removed %d rows as Age outliers", before - len(df))
            elif outlier_method == "clip":
                df["Age"] = np.clip(df["Age"], minimum_age, maximum_age)
                logging.info("Clipped Age values to IQR bounds")
            elif outlier_method == "flag":
                df["Age_outlier"] = ~df["Age"].between(minimum_age, maximum_age)
                logging.info("Flagged Age outliers in 'Age_outlier' column")
            else:
                raise ValueError("outlier_method must be one of 'remove','clip','flag'")

    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Create additional useful features and return the new DataFrame."""
    df = df.copy()
    # Ensure required numeric columns exist before arithmetic
    for col in ("SibSp", "Parch"):
        if col not in df.columns:
            df[col] = 0

    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = np.where(df["FamilySize"] == 1, 1, 0)
    # Protect against division by zero (FamilySize min is 1)
    df["FarePerPerson"] = (
        df["Fare"] / df["FamilySize"] if "Fare" in df.columns else np.nan
    )
    logging.info("Added features: FamilySize, IsAlone, FarePerPerson")
    return df


def plot_eda(df: pd.DataFrame, out_dir: Path, show: bool = False) -> None:
    """Save or show a few EDA plots into `out_dir`."""
    sns.set_style("whitegrid")
    out_dir.mkdir(parents=True, exist_ok=True)

    if "Age" in df.columns:
        plt.figure(figsize=(6, 4))
        df["Age"].hist()
        plt.title("Age Distribution")
        plt.xlabel("Age")
        plt.ylabel("Number of Passengers")
        path = out_dir / "age_distribution.png"
        plt.tight_layout()
        plt.savefig(path)
        logging.info("Saved %s", path)
        if show:
            plt.show()
        plt.close()

    if "Fare" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=df["Fare"])
        plt.title("Fare Distribution")
        path = out_dir / "fare_boxplot.png"
        plt.tight_layout()
        plt.savefig(path)
        logging.info("Saved %s", path)
        if show:
            plt.show()
        plt.close()

    numeric_columns = df.select_dtypes(include=np.number)
    if not numeric_columns.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_columns.corr(), annot=True)
        plt.title("Correlation Heatmap")
        path = out_dir / "correlation_heatmap.png"
        plt.tight_layout()
        plt.savefig(path)
        logging.info("Saved %s", path)
        if show:
            plt.show()
        plt.close()


def save_data(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    logging.info("Saved cleaned data to %s", path)


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple EDA and cleaning for Titanic dataset"
    )
    p.add_argument(
        "--input", "-i", default="titanic.txt", help="Path to input CSV/TSV file"
    )
    p.add_argument(
        "--output", "-o", default="Cleaned_Titanic.csv", help="Path to save cleaned CSV"
    )
    p.add_argument("--plots-dir", default="plots", help="Directory to save plots")
    p.add_argument(
        "--no-plots", action="store_true", help="Do not display plots interactively"
    )
    p.add_argument(
        "--outlier",
        choices=["remove", "clip", "flag"],
        default="remove",
        help="How to handle Age outliers",
    )
    p.add_argument("--sep", default=",", help="Separator for input file (default ',')")
    p.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    input_path = args.input
    output_path = args.output
    plots_dir = Path(args.plots_dir)

    # Load
    df = load_data(input_path, sep=args.sep)

    # Validate expected columns (best-effort)
    expected = ["Age", "Fare", "SibSp", "Parch", "Embarked"]
    try:
        validate_columns(df, expected)
    except ValueError as e:
        logging.warning("Validation warning: %s", e)

    # Clean
    df = clean_data(df, outlier_method=args.outlier)

    # Feature engineering
    df = feature_engineer(df)

    # EDA plots
    if args.no_plots:
        plot_eda(df, plots_dir, show=False)
    else:
        plot_eda(df, plots_dir, show=True)

    # Save
    save_data(df, output_path)
    logging.info("Project completed successfully")


if __name__ == "__main__":
    main()
