# 🚢 Titanic Survival Predictor — End-to-End ML Pipeline

A complete, production-style machine learning pipeline that takes raw Titanic passenger data and predicts survival — built entirely with `scikit-learn`'s `Pipeline` and `ColumnTransformer`, with no manual, error-prone preprocessing steps.

This isn't a "get accuracy once and stop" notebook. Every preprocessing decision below was reasoned through deliberately (including the mistakes caught and fixed along the way), then chained into a single reusable, leak-free pipeline object.

---

## 📊 Results

| Metric | Score |
|---|---|
| Test Accuracy (single split) | **80.4%** |
| 5-Fold Cross-Validation Mean Accuracy | **79.1%** |
| CV Fold Scores | `[0.771, 0.792, 0.792, 0.781, 0.820]` |

The tight spread across folds (77–82%) confirms the result is stable, not a lucky single split.

---

## 🧠 Preprocessing Decisions

Every column was individually evaluated — kept, engineered, or dropped — based on whether it was the target, a duplicate, a data leak, or genuinely useful signal.

| Column | Decision | Reasoning |
|---|---|---|
| `survived` | Target (`y`) | What we're predicting |
| `pclass` | Ordinal Encoding | Natural rank exists (1st > 2nd > 3rd class status) |
| `class` | **Dropped** | Duplicate of `pclass`, just as text labels |
| `sex` | One-Hot Encoding | Nominal, no natural order |
| `age` | Median Fill → Standardization | Missing values filled before scaling |
| `sibsp`, `parch` | **Dropped** | Used only to construct `family_size` |
| `family_size` | Engineered (`sibsp + parch + 1`) → Standardization | Captures family size more granularly than a binary flag |
| `alone` | **Dropped** | Redundant — `family_size` already captures this, with more detail |
| `fare` | Standardization | Continuous numeric feature |
| `embarked` | Mode Fill → One-Hot Encoding | Nominal, few missing values |
| `who`, `adult_male` | **Dropped** | Redundant with `sex` / `age` |
| `embark_town` | **Dropped** | Duplicate of `embarked` |
| `alive` | **Dropped** | Direct leak of the target — identical info to `survived` |
| `deck` | **Dropped** | ~77% missing values, not reliably imputable |

---

## 🏗️ Pipeline Architecture

```
Raw Titanic Data
      │
      ├── Feature Engineering: family_size = sibsp + parch + 1
      ├── Drop redundant/leaky columns
      ├── Train/Test Split (80/20, random_state=42)
      │
      └── ColumnTransformer
            ├── pclass       → OrdinalEncoder
            ├── sex          → OneHotEncoder
            ├── age          → SimpleImputer(median) → StandardScaler
            ├── embarked     → SimpleImputer(most_frequent) → OneHotEncoder
            ├── fare         → StandardScaler
            └── family_size  → StandardScaler
                  │
                  └── LogisticRegression
```

All of this is wrapped in a single `sklearn.pipeline.Pipeline` object — meaning the entire flow runs with one `.fit()` call and one `.predict()`/`.score()` call, with zero manual intervention and zero risk of train/test data leakage.

---

## 🛠️ Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn (`ColumnTransformer`, `Pipeline`, `OrdinalEncoder`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`, `LogisticRegression`)
- seaborn (built-in Titanic dataset)
- joblib (model persistence)

---

## 🚀 How to Run

```bash
# Clone the repo
git clone <your-repo-url>
cd <repo-folder>

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook end_to_end.ipynb
```

### Using the saved pipeline directly
```python
import joblib

pipe = joblib.load('titanic_pipeline.pkl')
prediction = pipe.predict(new_passenger_df)
```

`new_passenger_df` must contain the columns: `pclass`, `sex`, `age`, `fare`, `embarked`, `family_size`.

---

## 📁 Repo Structure

```
├── end_to_end.ipynb        # Full pipeline notebook
├── titanic_pipeline.pkl    # Saved, deployable pipeline
├── requirements.txt        # Dependencies
└── README.md
```

---

## 🔑 Key Takeaways

- **No data leakage:** `fit()` is only ever called on training data; `transform()` alone is used on test data.
- **No target leakage:** `alive` (a direct copy of `survived`) was identified and removed.
- **Validated, not assumed:** results were checked with 5-fold cross-validation, not just a single train/test split.
- **Deployable:** the entire pipeline — imputation, encoding, scaling, and model — is saved as one `.pkl` file, ready to be loaded and used without re-running any preprocessing code.

---

## 🔮 Possible Next Steps

- Try other models (Random Forest, XGBoost) and compare against Logistic Regression
- Hyperparameter tuning with `GridSearchCV`
- Deploy via a Streamlit app for interactive predictions
