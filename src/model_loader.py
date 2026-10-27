from functools import lru_cache
import joblib
import pandas as pd

from src.data_preprocessing import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "random_forest_pipeline.joblib"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "car_details_cleaned.csv"


@lru_cache(maxsize=1)
def load_model():
    """Load and cache the trained Random Forest pipeline."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_dataset():
    """Load and cache the cleaned car dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Cleaned dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)
