import pandas as pd
import numpy as np


def create_tenure_features(df: pd.DataFrame, snapshot_date: pd.Timestamp = None) -> pd.DataFrame:
    """
    Create tenure-based features.

    Active employees are measured up to the dataset snapshot date (the latest date in the data),
    not today's date: using today would add years to the tenure of active employees only,
    leak the target and make results change from run to run.
    """
    df = df.copy()

    if snapshot_date is None:
        snapshot_date = df[["DateofHire", "DateofTermination", "LastPerformanceReview_Date"]].max().max()
    df["EndDate"] = df["DateofTermination"].fillna(snapshot_date)

    df["TenureYears"] = (
        (df["EndDate"] - df["DateofHire"])
        .dt.days / 365
    ).clip(lower=0)

    df["TenureGroup"] = pd.cut(
        df["TenureYears"],
        bins=[0, 1, 3, 5, 10, 40],
        labels=["<1 year", "1–3 years", "3–5 years", "5–10 years", "10+ years"]
    )

    return df


def create_salary_features(df: pd.DataFrame, dept_avg_salary: pd.Series = None) -> tuple:
    """
    Create relative salary features.
    If dept_avg_salary is None, it's computed from this df (use only on train).
    Otherwise, the provided mapping (learned on train) is applied.
    """
    df = df.copy()
    if dept_avg_salary is None:
        dept_avg_salary = df.groupby("Department")["Salary"].mean()

    df["DeptAvgSalary"] = df["Department"].map(dept_avg_salary)
    # Departments unseen in the training data have no benchmark: treat as paid at par (1.0)
    df["RelativeSalary"] = (df["Salary"] / df["DeptAvgSalary"]).fillna(1.0)
    return df, dept_avg_salary


def create_engagement_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engagement and satisfaction risk flags.
    """
    df = df.copy()

    df["LowEngagementFlag"] = np.where(df["EngagementSurvey"] < 3.5, 1, 0)
    df["LowSatisfactionFlag"] = np.where(df["EmpSatisfaction"] <= 3, 1, 0)

    return df


def create_behavior_features(df: pd.DataFrame, absence_threshold: float = None) -> tuple:
    """
    Create attendance and behavior-based features.
    If absence_threshold is None, it's computed from this df (use only on train).
    """
    df = df.copy()
    if absence_threshold is None:
        absence_threshold = df["Absences"].median()

    df["HighAbsenceFlag"] = np.where(df["Absences"] > absence_threshold, 1, 0)
    df["LateRecentlyFlag"] = np.where(df["DaysLateLast30"] > 0, 1, 0)
    return df, absence_threshold


def encode_performance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode performance score as ordinal numeric feature.
    """
    df = df.copy()

    performance_map = {
        "PIP": 1,
        "Needs Improvement": 2,
        "Fully Meets": 3,
        "Exceeds": 4
    }

    df["PerformanceScoreNum"] = df["PerformanceScore"].map(performance_map)

    return df


def select_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select final feature set for modeling.
    """
    features = [
        "TenureYears",
        "RelativeSalary",
        "EngagementSurvey",
        "EmpSatisfaction",
        "SpecialProjectsCount",
        "DaysLateLast30",
        "Absences",
        "PerformanceScoreNum",
        "Attrition"
    ]

    return df[features]
