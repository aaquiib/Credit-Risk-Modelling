# Credit Risk Analyser

An end-to-end machine learning project that predicts credit risk for loan applicants using the German Credit Dataset. A trained XGBoost model is served through an interactive Streamlit web application that provides instant risk scores, probability breakdowns, and actionable recommendations.

**Live App:** https://credit-risk-modelling01.streamlit.app/
---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Model Performance](#model-performance)
- [Setup & Installation](#setup--installation)
- [Running the App](#running-the-app)
- [Notebook Workflow](#notebook-workflow)
- [Key Design Decisions](#key-design-decisions)

---

## Overview

Credit risk assessment is one of the most critical tasks in the financial industry. This project builds a complete ML pipeline from raw data to a deployed web application:

- **Exploratory Data Analysis** — distribution analysis, bivariate risk profiling, correlation study
- **Feature Engineering** — ordinal encoding for job skill level, one-hot encoding for nominal features, standard scaling for numerics
- **Model Comparison** — 9 classifiers benchmarked (Logistic Regression, Random Forest, SVM, Decision Tree, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Extra Trees)
- **Hyperparameter Tuning** — `RandomizedSearchCV` with 5-fold cross-validation optimising recall (minimising missed bad credit applicants)
- **Deployment** — interactive Streamlit app with a dark-themed UI and real-time predictions

---

## Live Demo

Run locally with:

```bash
streamlit run app.py
```

The app accepts applicant details (age, job level, housing, credit amount, loan duration, savings, checking account, purpose) and returns:

- Risk classification (Low Risk / High Risk)
- Good/Bad credit probability with visual progress bars
- Key factor chips highlighting risk signals
- A plain-language recommendation

---

## Dataset

**German Credit Data with Risk**
Sourced from [Kaggle — kabure/german-credit-data-with-risk](https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk)

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Rows | 1,000 applicants |
| Features | 9 (after preprocessing) |
| Target | `Risk` — `good` (700) / `bad` (300) |
| Missing values | `Saving accounts` (183), `Checking account` (394) — handled by row removal |

**Feature descriptions:**

| Feature | Type | Description |
|---|---|---|
| Age | Numerical | Applicant age in years |
| Sex | Nominal | `male` / `female` |
| Job | Ordinal | Skill level: 0 (unskilled non-resident) → 3 (highly skilled) |
| Housing | Nominal | `own` / `rent` / `free` |
| Saving accounts | Nominal | `little` / `moderate` / `quite rich` / `rich` |
| Checking account | Nominal | `little` / `moderate` / `rich` |
| Credit amount | Numerical | Loan amount in Deutsche Marks |
| Duration | Numerical | Loan duration in months |
| Purpose | Nominal | Reason for the loan (car, education, furniture, etc.) |

---

## Project Structure

```
credit-risk-modelling/
│
├── data/
│   └── german_credit_data.csv        # Raw dataset
│
├── notebook/
│   ├── 01_eda.ipynb                  # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Preprocessing pipeline definition
│   ├── 03_model_training.ipynb       # Multi-model benchmarking
│   ├── 04_hyperparameter_tuning.ipynb# RandomizedSearchCV + final model
│   ├── saved_models/
│   │   └── Xgboost_model1.joblib     # Serialised final pipeline
│   └── archive/
│       └── analysis.ipynb            # Original monolithic notebook
│
├── src/
│   ├── __init__.py
│   └── preprocessing.py              # load_and_clean(), build_preprocessor()
│
├── app.py                            # Streamlit web application
├── requirements.txt                  # Pinned dependencies
└── .gitignore
```

---

## Tech Stack

| Layer | Library | Version |
|---|---|---|
| Data manipulation | pandas | 2.3.3 |
| Numerical computing | numpy | 2.4.2 |
| Machine learning | scikit-learn | 1.8.0 |
| Gradient boosting | xgboost | 3.2.0 |
| Gradient boosting | lightgbm | 4.6.0 |
| Visualisation | matplotlib | 3.10.8 |
| Visualisation | seaborn | 0.13.2 |
| Model serialisation | joblib | 1.5.3 |
| Web application | streamlit | 1.54.0 |

---

## Model Performance

The final model is an XGBoost classifier with hyperparameters tuned via `RandomizedSearchCV` (50 iterations, 5-fold CV, optimising recall).

| Metric | Score |
|---|---|
| Accuracy | 74.3% |
| Precision | 75.0% |
| **Recall** | **81.4%** |
| F1 Score | 78.0% |
| ROC-AUC | 75.9% |

Recall was chosen as the primary optimisation metric because missing a bad credit applicant (false negative) is more costly to a lender than incorrectly flagging a good applicant (false positive).

**Final XGBoost hyperparameters:**

| Parameter | Value |
|---|---|
| `n_estimators` | 200 |
| `learning_rate` | 0.03 |
| `max_depth` | 4 |
| `min_child_weight` | 4 |
| `subsample` | 0.911 |
| `colsample_bytree` | 0.689 |
| `reg_alpha` | 0.333 |
| `reg_lambda` | 1.0 |

---

## Setup & Installation

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone <repo-url>
cd credit-risk-modelling

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

> The app loads the pre-trained model from `notebook/saved_models/Xgboost_model1.joblib`.
> If you retrain the model, re-run notebook `04_hyperparameter_tuning.ipynb` to regenerate this file.

---

## Notebook Workflow

The notebooks are designed to be run in order from the `notebook/` directory:

| Notebook | Purpose |
|---|---|
| `01_eda.ipynb` | Load data, handle missing values, univariate and bivariate analysis, risk profiling |
| `02_feature_engineering.ipynb` | Define feature types, build the preprocessing pipeline, verify train/test split |
| `03_model_training.ipynb` | Train 9 classifiers inside sklearn Pipelines, compare accuracy/recall/F1/AUC |
| `04_hyperparameter_tuning.ipynb` | Tune top models with `RandomizedSearchCV`, evaluate the best, save the final pipeline |

All notebooks import shared utilities via:

```python
import sys
sys.path.insert(0, '..')
from src.preprocessing import load_and_clean, build_preprocessor
```

---

## Key Design Decisions

**Drop vs. impute missing values**
`Saving accounts` and `Checking account` each have significant missingness (18% and 39% respectively). Analysis showed that NaN rows had a noticeably lower bad-credit rate (~17%) compared to the `little` category (~49%), suggesting the missing values are not random. Dropping these rows retains a clean, reliable signal at the cost of reducing the dataset from 1,000 to 522 samples. Imputation with a constant category is a viable alternative if a larger training set is a priority.

**Optimising recall over accuracy**
In a credit risk context, the cost of approving a bad borrower (missed bad credit) typically exceeds the cost of rejecting a good one. `RandomizedSearchCV` was therefore configured with `scoring='recall'` to find models that minimise false negatives.

**Full pipeline serialisation**
The entire sklearn `Pipeline` (preprocessor + model) is saved as a single `.joblib` file. The Streamlit app loads this pipeline and passes raw, unprocessed input directly to it — no preprocessing logic is duplicated in `app.py`.
