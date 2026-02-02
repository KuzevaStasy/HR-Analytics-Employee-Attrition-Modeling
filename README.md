# HR Analytics – Employee Attrition Modeling

End-to-end People Analytics project focused on understanding and predicting
employee attrition using interpretable machine learning techniques.

The project demonstrates how HR data can be transformed into actionable insights
that support retention strategies and data-informed decision-making.

---

## 📌 Project Overview

Employee attrition is a costly and persistent challenge for organizations.
This project explores the key drivers behind employee turnover and builds
predictive models to estimate attrition risk at the individual level.

The emphasis is placed on:
- explainability over black-box performance
- business relevance over pure technical optimization
- clean, reusable code over notebook-only analysis

---

## 🎯 Business Objectives

- Identify factors associated with employee attrition
- Quantify attrition risk using interpretable models
- Demonstrate how HR teams could proactively prioritize retention efforts
- Provide a reusable analytics pipeline suitable for extension or deployment

---

## 📂 Project Structure

HR-Analytics-Employee-Insights/

│

├── data/

│ ├── raw/ # Original HR dataset

│ └── processed/ # Model-ready datasets

│

├── notebooks/

│ ├── 01_EDA_HR_Insights.ipynb

│ ├── 02_Feature_Engineering.ipynb

│ └── 03_Attrition_Modeling.ipynb

│

├── src/

│ ├── preprocessing.py # Data loading and cleaning

│ ├── features.py # Feature engineering logic

│ └── modeling.py # Training and evaluation utilities

│

├── main.py # End-to-end pipeline entry point

├── requirements.txt

└── README.md


---

## 📊 Key Analytical Insights

- Attrition risk is highest among employees with shorter tenure
- Engagement and satisfaction are stronger predictors than performance alone
- Relative (department-level) salary matters more than absolute pay
- High performers are not immune to attrition if engagement is low
- Attendance-related signals provide secondary but complementary information

These findings align with real-world HR research and reinforce the importance
of early intervention and engagement-focused retention strategies.

---

## 🧠 Modeling Approach

Two models are explored:

### Logistic Regression (Primary Model)
- High interpretability
- Clear directional impact of features
- Suitable for HR-facing decision support

### Random Forest (Benchmark)
- Captures non-linear relationships
- Used for comparison and validation

Model evaluation is based on ROC AUC, with stratified train-test splits to
handle class imbalance.

---

## 📈 Attrition Risk Scoring

As a practical output, the pipeline produces an **attrition risk score**
(probability between 0 and 1) for each employee.

These scores can be used to:
- identify high-risk employees
- prioritize retention initiatives
- support HR business partner discussions

Predictions are intended as **decision-support tools**, not automated actions.

---

## ⚠️ Limitations & Responsible Use

- The dataset is relatively small and may not generalize across organizations
- Reported performance metrics represent an upper bound, not production results
- Attrition risk predictions should always be complemented with human judgment
- Ethical use of employee data and transparency are essential

---

## ▶️ How to Run the Project

1. Install dependencies:
```bash
pip install -r requirements.txt

2. Run the full pipeline:
```bash
python main.py
