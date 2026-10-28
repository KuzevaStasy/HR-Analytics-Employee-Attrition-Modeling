# HR Analytics – Employee Attrition Modeling

End-to-end People Analytics project focused on understanding and predicting
employee attrition using interpretable machine learning techniques.

The project shows how HR data can be turned into insights for retention strategies,
and how to check honestly what the data can and cannot support.

---

## 📌 Project Overview

Employee attrition is a costly and persistent challenge for organizations.
This project explores the factors associated with employee turnover and builds
predictive models to estimate attrition risk at the individual level.

The emphasis is placed on:
- explainability over black-box performance
- business relevance over pure technical optimization
- leakage-safe feature engineering (the train/test split comes before any learned statistic)
- clean, reusable code over notebook-only analysis

---

## 🎯 Business Objectives

- Identify factors associated with employee attrition
- Quantify attrition risk using interpretable models
- Demonstrate how HR teams could prioritize retention efforts
- Provide a reusable analytics pipeline suitable for extension

---

## 📂 Project Structure

```
HR-Analytics-Employee-Attrition-Modeling/
├── data/
│   ├── raw/
│   │   └── HRDataset_v14.csv                 # Original HR dataset (311 employees)
│   └── processed/
│       ├── hr_model_ready_train.csv          # Model-ready training set (from notebook 02)
│       └── hr_model_ready_test.csv           # Model-ready test set (from notebook 02)
├── notebooks/
│   ├── 01_EDA_HR_Insights.ipynb              # Exploratory analysis
│   ├── 02_Feature_Engineering.ipynb          # Features and train/test split
│   └── 03_Attrition_Modeling.ipynb           # Models, evaluation and risk scoring
├── src/
│   ├── preprocessing.py                      # Data loading and cleaning
│   ├── features.py                           # Feature engineering logic
│   └── modeling.py                           # Training and evaluation utilities
├── main.py                                   # End-to-end pipeline entry point
├── requirements.txt
└── README.md
```

---

## 🗂️ Data

`data/raw/HRDataset_v14.csv` contains 311 employees and 36 columns (demographics, position, salary, performance, engagement, absences and employment dates).

**Attrition definition:** an employee counts as attrition if `EmploymentStatus` is not *Active*. That is 104 of 311 employees (33%): 88 *Voluntarily Terminated* and 16 *Terminated for Cause*.

---

## 📊 Key Analytical Insights

From the exploratory analysis (notebook 01):

- About one-third of employees left; 28% left voluntarily.
- Attrition is concentrated in Production (about 40% of its 209 employees). Software Engineering looks similar (36%) but has only 11 employees, so that figure is uncertain.
- Satisfaction and engagement survey scores do **not** differ between employees who left and those who stayed.
- Leavers earn slightly less, but the difference is not statistically significant.
- Attrition occurs at every performance level (22% among top performers, 56% for *Needs Improvement*, a small group).
- Absences differ little, and recent lateness is only slightly higher among leavers.

From the modeling (notebook 03):

- Tenure is the only feature with a clear signal: shorter tenure goes with higher attrition risk.
- Without tenure, the models are close to random guessing (ROC AUC 0.52 - 0.59), so engagement, satisfaction, salary, performance and attendance add little in this dataset.

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

Evaluation uses ROC AUC on a stratified 75/25 train-test split, plus 5-fold cross-validation because the test set is small (78 employees).

| Model | Test ROC AUC | 5-fold CV ROC AUC |
|---|---|---|
| Logistic Regression | 0.82 | 0.79 |
| Random Forest | 0.77 | 0.75 |

### Leakage-safe feature engineering

- The train/test split is done **before** any statistic is learned (department average salary, absence median), so nothing from the test set leaks into the training features.
- Tenure is measured to the dataset snapshot date (28 Feb 2019) for active employees and to the termination date for leavers. Using today's date instead would inflate the tenure of active employees only and leak the target.
- Departments that do not appear in the training set get a relative salary of 1.0 (paid at par).

---

## 📈 Attrition Risk Scoring

The pipeline produces an **attrition risk score** (probability between 0 and 1) for each employee in the test set.
On the test set, the observed attrition rate rises with the predicted risk: about 12% in the low-risk group, 19% in the medium group and 69% in the high-risk group.

These scores can be used to rank employees by relative risk and to support HR business partner discussions.
Predictions are intended as **decision-support tools**, not automated actions.

---

## ⚠️ Limitations & Responsible Use

- The dataset is small (311 employees) and may not generalize across organizations; test-set metrics are noisy.
- Almost all predictive power comes from tenure, which is partly determined by how the data was recorded (leavers are measured to their exit date, active employees to the snapshot date). The results are best read as a demonstration of the workflow, not as a validated retention tool.
- Attrition includes involuntary exits (terminated for cause), which are not retention-driven.
- Attrition risk predictions should always be complemented with human judgment
- Ethical use of employee data and transparency are essential

---

## ▶️ How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/KuzevaStasy/HR-Analytics-Employee-Attrition-Modeling.git
cd HR-Analytics-Employee-Attrition-Modeling
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the analysis

Run the notebooks in order (`01` → `02` → `03`); notebook 02 creates the train/test files that notebook 03 reads.
```bash
jupyter notebook notebooks/
```

Or run the end-to-end pipeline from the command line (trains the logistic regression, prints the test ROC AUC and the five highest-risk employees in the test set):
```bash
python main.py
```

---

## 💻 Tech Stack

- Python 3.10+
- Data: pandas, numpy, scipy
- Modeling: scikit-learn
- Visualization: matplotlib, seaborn
- Environment: Jupyter Notebooks
