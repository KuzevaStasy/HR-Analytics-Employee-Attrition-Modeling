import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def split_data(df: pd.DataFrame, target: str = "Attrition"):
    """
    Split dataset into train and test sets.
    """
    X = df.drop(columns=target)
    y = df[target]

    return train_test_split(
        X, y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )


def train_logistic_regression(X_train, y_train):
    """
    Train logistic regression model with scaling.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler


def train_random_forest(X_train, y_train):
    """
    Train random forest classifier.
    """
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )

    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, scaler=None):
    """
    Evaluate model using ROC AUC.
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_prob)
