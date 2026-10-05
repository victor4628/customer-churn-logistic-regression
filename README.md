# Customer churn logistic regression

This project completes the 12 homework tasks using `Mocked_Customer_Data_With_Missing.xlsx`.

## Run

Python 3.12 was used. From the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe analysis.py
```

Open `outputs/report.html` for the self-contained brief report, or `customer_churn_analysis.ipynb` for the executed notebook. `analysis.py` contains the full reproducible workflow, organized into the assignment's steps. The original workbook is preserved.

The target is `churn_flag`, corresponding to `churn` in the homework. The workflow excludes customer identifiers, splits 80/20 with stratification, fits imputation inside training/CV folds, compares three logistic regression pipelines on training-only 5-fold CV, and evaluates the selected pipeline on held-out validation data. Probability threshold is 0.5; churn=1 is the positive class.

CSV evidence and `results.json` are in `outputs/`. The imputed CSV uses only training-fit medians/modes. `churn_model.joblib` stores the selected pipeline with preprocessing; load serialized models only from trusted sources and use matching library versions.
