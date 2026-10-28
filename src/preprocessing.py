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
    df = pd.read_csv(path)
    # Some text columns carry trailing spaces (e.g. 'Production       ')
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].str.strip()
    return df


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse date columns into datetime format.
    """
    df = df.copy()
    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], format="%m/%d/%Y", errors="coerce")
    return df


TERMINATED_STATUSES = ["Voluntarily Terminated", "Terminated for Cause"]

def create_attrition_target(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Attrition"] = np.where(df["EmploymentStatus"].isin(TERMINATED_STATUSES), 1, 0)
    return df
