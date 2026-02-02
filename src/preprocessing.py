import pandas as pd
import numpy as np


DATE_COLS = [
    "DateofHire",
    "DateofTermination",
    "LastPerformanceReview_Date"
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load raw HR dataset.
    """
    return pd.read_csv(path)


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns into datetime format.
    """
    df = df.copy()
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def create_attrition_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary attrition target.
    Active employees -> 0
    Terminated employees -> 1
    """
    df = df.copy()
    df["Attrition"] = np.where(df["EmploymentStatus"] == "Active", 0, 1)
    return df
