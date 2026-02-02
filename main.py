import logging

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
    split_data,
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

    # 2. Preprocessing
    logging.info("Parsing date columns")
    df = parse_dates(df)

    logging.info("Creating attrition target")
    df = create_attrition_target(df)

    # 3. Feature engineering
    logging.info("Creating tenure features")
    df = create_tenure_features(df)

    logging.info("Creating salary features")
    df = create_salary_features(df)

    logging.info("Creating engagement and satisfaction features")
    df = create_engagement_features(df)

    logging.info("Creating behavioral features")
    df = create_behavior_features(df)

    logging.info("Encoding performance scores")
    df = encode_performance(df)

    # 4. Select modeling features
    logging.info("Selecting model features")
    df_model = select_model_features(df).copy()

    # 5. Train / test split
    logging.info("Splitting data into train and test sets")
    X_train, X_test, y_train, y_test = split_data(df_model)

    # 6. Train model
    logging.info("Training logistic regression model")
    model, scaler = train_logistic_regression(X_train, y_train)

    # 7. Evaluate model
    logging.info("Evaluating model performance")
    auc = evaluate_model(model, X_test, y_test, scaler)

    logging.info(f"ROC AUC: {auc:.3f}")

    logging.info(
        "Note: Perfect or near-perfect performance is likely due to the "
        "small dataset size and should be interpreted with caution."
    )

    # 8. Example business output: attrition risk scores
    logging.info("Calculating attrition risk scores for all employees")

    X_all = df_model.drop(columns="Attrition")
    X_all_scaled = scaler.transform(X_all)

    df_model["AttritionRisk"] = model.predict_proba(X_all_scaled)[:, 1]

    top_risk = (
        df_model
        .sort_values("AttritionRisk", ascending=False)
        .head(5)[["AttritionRisk"]]
    )

    logging.info("Top 5 highest-risk employees (by model score):")
    logging.info(f"\n{top_risk}")

    logging.info("Pipeline finished successfully")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()
