"""
This module will load the raw dataset, standardize column names,
validate data quality, and prepare a cleaned dataset for modeling.
"""

from pathlib import Path
import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "car_details.csv"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
CLEANED_DATA_PATH = PROCESSED_DATA_DIR / "car_details_cleaned.csv"


def load_raw_data(file_path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the original used-car dataset.
    Parameters:
    file_path : Path
        Location of the raw CSV file.
    Returns
        A DataFrame containing the raw dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at: {file_path}"
        )

    return pd.read_csv(file_path)



def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to lowercase snake_case.
    Examples:
        'Fuel Type' -> 'fuel_type'
        'Max Power' -> 'max_power'
        'Seating Capacity' -> 'seating_capacity'
    Returns a copy so the original DataFrame is not modified.
    """
    cleaned_df = df.copy()
    cleaned_df.columns = (
        cleaned_df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    if cleaned_df.columns.duplicated().any():
        duplicate_columns = cleaned_df.columns[
            cleaned_df.columns.duplicated()
        ].tolist()
        raise ValueError(
            f"Duplicate column names after standardization: "
            f"{duplicate_columns}"
        )

    return cleaned_df



def generate_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate a data-quality report for every column.

    Reports data type, missing values, missing percentage,
    and number of distinct non-missing values.
    """
    report = pd.DataFrame({
        "data_type": df.dtypes.astype(str),
        "missing_count": df.isnull().sum(),
        "missing_percentage": (
            df.isnull().mean() * 100
        ).round(2),
        "unique_values": df.nunique(dropna=True),
    })

    return report


def validate_required_columns(df: pd.DataFrame) -> None:
    """
    Ensure essential columns required for price prediction exist.
    """
    required_columns = {
        "make",
        "model",
        "price",
        "year",
        "kilometer",
    }

    missing_columns = required_columns.difference(df.columns)

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{sorted(missing_columns)}"
        )



def clean_text_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove unnecessary whitespace from text columns.
    Empty strings are converted to missing values.
    Meaningful categorical values are preserved.
    """
    cleaned_df = df.copy()

    text_columns = cleaned_df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        cleaned_df[column] = (
            cleaned_df[column]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    return cleaned_df



def clean_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """
    Convert specified columns to numeric values.

    Values that cannot be converted become missing values.
    The input DataFrame is not modified.
    """
    cleaned_df = df.copy()

    for column in columns:
        if column not in cleaned_df.columns:
            raise ValueError(
                f"Column '{column}' does not exist in the dataset."
            )

        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        )

    return cleaned_df



def validate_numeric_values(df: pd.DataFrame) -> dict:
    """
    Report potentially invalid values in core numeric columns.
    This function does not modify or delete records.
    """
    required_columns = ["price", "year", "kilometer"]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: {missing_columns}"
        )

    current_year = pd.Timestamp.now().year

    report = {
        "missing_price": int(df["price"].isna().sum()),
        "non_positive_price": int(
            (df["price"] <= 0).fillna(False).sum()
        ),
        "missing_year": int(df["year"].isna().sum()),
        "year_before_1900": int(
            (df["year"] < 1900).fillna(False).sum()
        ),
        "year_after_current_year": int(
            (df["year"] > current_year).fillna(False).sum()
        ),
        "missing_kilometer": int(df["kilometer"].isna().sum()),
        "negative_kilometer": int(
            (df["kilometer"] < 0).fillna(False).sum()
        ),
    }

    return report



def extract_specification_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the leading numeric value from engine, max_power,
    and max_torque while preserving missing values.
    """
    cleaned_df = df.copy()

    specification_columns = [
        "engine",
        "max_power",
        "max_torque",
    ]

    for column in specification_columns:
        if column not in cleaned_df.columns:
            raise ValueError(
                f"Required specification column missing: {column}"
            )

        numeric_values = (
            cleaned_df[column]
            .astype("string")
            .str.extract(
                r"^\s*([-+]?\d*\.?\d+)",
                expand=False,
            )
        )

        cleaned_df[column] = pd.to_numeric(
            numeric_values,
            errors="coerce",
        )

    return cleaned_df



def convert_measurement_columns(df: pd.DataFrame,) -> pd.DataFrame:
    """
    Ensure vehicle measurement columns use numeric data types.
    Invalid values become missing values; existing missing values
    remain missing.
    """
    cleaned_df = df.copy()

    measurement_columns = [
        "length",
        "width",
        "height",
        "seating_capacity",
        "fuel_tank_capacity",
    ]

    for column in measurement_columns:
        if column not in cleaned_df.columns:
            raise ValueError(
                f"Required measurement column missing: {column}"
            )

        cleaned_df[column] = pd.to_numeric(
            cleaned_df[column],
            errors="coerce",
        )

    return cleaned_df



def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the documented cleaning steps to the raw dataset.
    Missing values are preserved for later treatment in the
    model-training pipeline.
    """
    cleaned_df = standardize_column_names(df)
    cleaned_df = clean_text_values(cleaned_df)
    validate_required_columns(cleaned_df)

    cleaned_df = clean_numeric_columns(
        cleaned_df,
        columns=["price", "year", "kilometer"],
    )

    cleaned_df = extract_specification_values(cleaned_df)
    cleaned_df = convert_measurement_columns(cleaned_df)

    return cleaned_df


def save_cleaned_data(df: pd.DataFrame) -> Path:
    """
    Save the cleaned dataset to the processed-data directory.
    The original raw dataset is not modified.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(CLEANED_DATA_PATH, index=False)

    return CLEANED_DATA_PATH
