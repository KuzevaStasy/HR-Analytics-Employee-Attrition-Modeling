import logging

from sklearn.model_selection import train_test_split

from src.preprocessing import (
    load_data,
    parse_dates,
    create_attrition_target
)
from src.features import (
    create_tenure_features,
    create_salary_features,
    create_engagement_features,
    create_behavior_features,
    encode_performance,
    select_model_features
)
from src.modeling import (
    train_logistic_regression,
    evaluate_model
)

# -----------------------------------------------------------------------------
# Logging configuration
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def run_pipeline() -> None:
    logging.info("Starting HR attrition modeling pipeline")

    # 1. Load raw data
    logging.info("Loading raw dataset")
    df = load_data("data/raw/HRDataset_v14.csv")

    # 2. Preprocessing (безопасно за целия df — не зависи от train/test)
    logging.info("Parsing date columns")
    df = parse_dates(df)

    logging.info("Creating attrition target")
    df = create_attrition_target(df)

    logging.info("Creating tenure features")
    df = create_tenure_features(df)

    logging.info("Encoding performance scores")
    df = encode_performance(df)

    # 3. Train / test split — ПРЕДИ да смятаме каквато и да е агрегатна статистика
    logging.info("Splitting data into train and test sets")
    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["Attrition"]
    )

    # 4. Feature engineering — статистиките се учат само от train
    logging.info("Creating salary features (fit on train)")
    train_df, dept_avg_salary = create_salary_features(train_df)
    test_df, _ = create_salary_features(test_df, dept_avg_salary=dept_avg_salary)

    logging.info("Creating behavioral features (fit on train)")
    train_df, absence_threshold = create_behavior_features(train_df)
    test_df, _ = create_behavior_features(test_df, absence_threshold=absence_threshold)

    logging.info("Creating engagement and satisfaction features")
    train_df = create_engagement_features(train_df)
    test_df = create_engagement_features(test_df)

    # 5. Избор на финални колони за модела
    logging.info("Selecting model features")
    train_model = select_model_features(train_df).copy()
    test_model = select_model_features(test_df).copy()

    X_train = train_model.drop(columns="Attrition")
    y_train = train_model["Attrition"]
    X_test = test_model.drop(columns="Attrition")
    y_test = test_model["Attrition"]

    # 6. Train model
    logging.info("Training logistic regression model")
    model, scaler = train_logistic_regression(X_train, y_train)

    # 7. Evaluate model
    logging.info("Evaluating model performance")
    auc = evaluate_model(model, X_test, y_test, scaler)
    logging.info(f"ROC AUC: {auc:.3f}")

    # 8. Risk scores — само върху test set (не смесваме train/test при "прогнозата")
    logging.info("Calculating attrition risk scores for test employees")
    X_test_scaled = scaler.transform(X_test)
    test_model["AttritionRisk"] = model.predict_proba(X_test_scaled)[:, 1]

    top_risk = (
        test_model
        .sort_values("AttritionRisk", ascending=False)
        .head(5)[["AttritionRisk"]]
    )
    logging.info("Top 5 highest-risk employees (test set, by model score):")
    logging.info(f"\n{top_risk}")

    logging.info("Pipeline finished successfully")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
