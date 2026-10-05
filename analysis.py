"""Customer churn homework. Run from the repository: python analysis.py."""
# %% 1. Load the workbook into a data frame
from pathlib import Path
import base64
import hashlib
import json
import platform
import warnings
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
    precision_recall_curve, log_loss)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PowerTransformer, PolynomialFeatures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('error', category=ConvergenceWarning)
ROOT = Path.cwd()
SOURCE = ROOT / 'Mocked_Customer_Data_With_Missing.xlsx'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SEED = 42
GREEN = '#254C45'
SECOND = '#B77744'
plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
    'axes.titlecolor': GREEN, 'axes.labelcolor': '#333333', 'font.size': 11,
    'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
excel = pd.ExcelFile(SOURCE)
assert len(excel.sheet_names) == 1, 'Review multiple sheets before choosing the source.'
df = pd.read_excel(excel)
print('Source sheet:', excel.sheet_names[0], '| Shape:', df.shape)

# %% 2. Completeness, variable types and data quality
TARGET = 'churn_flag'  # The screenshot calls this variable "churn".
CAT = ['gender', 'marrital', 'home_owner', 'fortune']
NUM = [c for c in df.columns if c not in CAT + [TARGET, 'cust_id']]
assert df[TARGET].notna().all() and set(df[TARGET].unique()) == {0, 1}
assert df.cust_id.is_unique, 'Repeated customers require a grouped split.'
assert np.isfinite(df[NUM].dropna().to_numpy()).all()
quality = pd.DataFrame({'variable': df.columns, 'dtype': df.dtypes.astype(str).values,
    'role': ['identifier' if c == 'cust_id' else 'binary target' if c == TARGET
             else 'categorical' if c in CAT else 'numeric' for c in df.columns],
    'missing_n': df.isna().sum().values, 'missing_pct': df.isna().mean().values * 100,
    'unique_n': df.nunique().values})
quality.to_csv(OUT / 'data_quality.csv', index=False)
df[NUM].describe().T.to_csv(OUT / 'numeric_summary.csv')
class_counts = df[TARGET].value_counts().sort_index()
print(quality.to_string(index=False))
print('Class counts:', class_counts.to_dict(), '| Duplicate rows:', df.duplicated().sum())
print('Categorical levels:', {c: sorted(df[c].dropna().unique().tolist()) for c in CAT})

# %% 3. Age histograms grouped by churn, using observed ages
fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), sharex=True, sharey=True)
bins = np.linspace(df.age.min(), df.age.max(), 21)
age_summary = df.groupby(TARGET).age.agg(['count', 'mean', 'median', 'std'])
age_summary.to_csv(OUT / 'age_by_churn.csv')
for label, ax, color in zip([0, 1], axes, [GREEN, SECOND]):
    ages = df.loc[df[TARGET] == label, 'age'].dropna()
    ax.hist(ages, bins=bins, density=True, color=color, alpha=.85, edgecolor='white')
    ax.set(title=f'Churn = {label} (observed n = {len(ages):,})', xlabel='Age')
axes[0].set_ylabel('Density')
fig.tight_layout()
fig.savefig(OUT / 'age_histograms.png', dpi=180)
plt.close(fig)
print(age_summary)

# %% 4. Welch two-sided t-test of observed total bills
bill0 = df.loc[df[TARGET] == 0, 'tot_bill'].dropna().to_numpy()
bill1 = df.loc[df[TARGET] == 1, 'tot_bill'].dropna().to_numpy()
t_result = stats.ttest_ind(bill1, bill0, equal_var=False)
diff = bill1.mean() - bill0.mean()
se = np.sqrt(bill1.var(ddof=1)/len(bill1) + bill0.var(ddof=1)/len(bill0))
ci = diff + np.array([-1, 1]) * stats.t.ppf(.975, t_result.df) * se
pooled_sd = np.sqrt(((len(bill1)-1)*bill1.var(ddof=1) +
    (len(bill0)-1)*bill0.var(ddof=1))/(len(bill1)+len(bill0)-2))
ttest = {'n_churn_0': len(bill0), 'n_churn_1': len(bill1),
    'mean_churn_0': float(bill0.mean()), 'mean_churn_1': float(bill1.mean()),
    'difference_1_minus_0': float(diff), 't': float(t_result.statistic),
    'df': float(t_result.df), 'p_two_sided': float(t_result.pvalue),
    'ci95_low': float(ci[0]), 'ci95_high': float(ci[1]), 'cohens_d': float(diff / pooled_sd)}
p_text = 'p < 0.001 (the computed p-value underflows to 0; it is not literally zero)' if t_result.pvalue == 0 else f'p = {t_result.pvalue:.6g}'
print(json.dumps(ttest, indent=2))

# %% 5. Missing-value strategy (define now, fit after splitting)
# Numeric: training median. Categorical: training mode. Never impute the target.
# Preserve binary home_owner as categorical despite its integer storage type.
# Exclude cust_id. num_cont, tot_call and num_out are numeric count variables.
X = df.drop(columns=[TARGET, 'cust_id']).copy()
y = df[TARGET].copy()
for c in CAT:
    X[c] = X[c].astype(object)
categorical_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
    ('encode', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))])
numeric_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())])
print('Numeric predictors:', NUM, '\nCategorical predictors:', CAT)

# %% 6. Stratified training/validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2,
    stratify=y, random_state=SEED)
assert set(X_train.index).isdisjoint(X_val.index)
split = pd.Series('training', index=df.index, name='split')
split.loc[X_val.index] = 'validation'
pd.DataFrame({'cust_id': df.cust_id, 'split': split}).to_csv(OUT / 'split_assignment.csv', index=False)
print('Training:', len(y_train), '| Validation:', len(y_val),
    '| Churn rates:', y_train.mean(), y_val.mean())

# %% 7. Baseline logistic regression with regularization
preprocess = ColumnTransformer([('numeric', numeric_pipe, NUM),
    ('categorical', categorical_pipe, CAT)], verbose_feature_names_out=False)
baseline = Pipeline([('preprocess', preprocess),
    ('model', LogisticRegression(C=1.0, max_iter=2000, solver='lbfgs', random_state=SEED))])

# %% 8. Training-only 5-fold cross-validation and transformation comparison
# Same folds for every candidate. The validation set is untouched during selection.
power_num = Pipeline([('imputer', SimpleImputer(strategy='median')),
    ('power', PowerTransformer(method='yeo-johnson', standardize=True))])
power = Pipeline([('preprocess', ColumnTransformer([
    ('numeric', power_num, NUM), ('categorical', categorical_pipe, CAT)],
    verbose_feature_names_out=False)), ('model', clone(baseline.named_steps['model']))])
pair = ['age', 'tot_bill']
quad_num = Pipeline([('imputer', SimpleImputer(strategy='median')),
    ('scale_before', StandardScaler()),
    ('quadratic', PolynomialFeatures(degree=2, include_bias=False)),
    ('scale_after', StandardScaler())])
quadratic = Pipeline([('preprocess', ColumnTransformer([
    ('quadratic', quad_num, pair), ('numeric', numeric_pipe, [c for c in NUM if c not in pair]),
    ('categorical', categorical_pipe, CAT)], verbose_feature_names_out=False)),
    ('model', clone(baseline.named_steps['model']))])
candidates = {'baseline': baseline, 'yeo_johnson': power, 'age_bill_quadratic': quadratic}
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {'roc_auc': 'roc_auc', 'average_precision': 'average_precision',
    'precision': 'precision', 'recall': 'recall', 'f1': 'f1', 'accuracy': 'accuracy'}
fold_rows = []
for name, model in candidates.items():
    scores = cross_validate(model, X_train, y_train, cv=folds, scoring=scoring,
        n_jobs=1, error_score='raise')
    for k in range(5):
        fold_rows.append({'model': name, 'fold': k+1,
            **{metric: float(scores['test_'+metric][k]) for metric in scoring}})
    print(name, 'CV ROC AUC:', scores['test_roc_auc'].mean(), flush=True)
cv_detail = pd.DataFrame(fold_rows)
cv_summary = cv_detail.groupby('model')[list(scoring)].agg(['mean', 'std'])
cv_detail.to_csv(OUT / 'cross_validation_folds.csv', index=False)
cv_summary.to_csv(OUT / 'cross_validation_summary.csv')
best_name = cv_summary[('roc_auc', 'mean')].idxmax()
selected = clone(candidates[best_name]).fit(X_train, y_train)
baseline.fit(X_train, y_train)
print('Selected by training CV ROC AUC:', best_name)

# %% 9. Interpret the baseline coefficients and save filled data
# Numeric odds ratios are per one training standard deviation, conditional on other predictors.
# Categorical coefficients compare against the first (dropped) category.
names = baseline.named_steps['preprocess'].get_feature_names_out()
coefficients = pd.DataFrame({'feature': names,
    'coefficient': baseline.named_steps['model'].coef_[0]})
coefficients['odds_ratio'] = np.exp(coefficients.coefficient)
coefficients['abs_coefficient'] = coefficients.coefficient.abs()
coefficients = coefficients.sort_values('abs_coefficient', ascending=False)
coefficients.to_csv(OUT / 'baseline_coefficients.csv', index=False)
encoder = baseline.named_steps['preprocess'].named_transformers_['categorical'].named_steps['encode']
references = {c: str(levels[0]) for c, levels in zip(CAT, encoder.categories_)}
filled = df.copy()
num_imputer = baseline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['imputer']
cat_imputer = baseline.named_steps['preprocess'].named_transformers_['categorical'].named_steps['imputer']
filled[NUM] = num_imputer.transform(X[NUM])
filled[CAT] = cat_imputer.transform(X[CAT])
assert filled.isna().sum().sum() == 0
filled.to_csv(OUT / 'customer_data_imputed.csv', index=False)
print(coefficients.head(10).to_string(index=False))
print('Categorical reference levels:', references)
selected_coefficients = pd.DataFrame({
    'feature': selected.named_steps['preprocess'].get_feature_names_out(),
    'coefficient': selected.named_steps['model'].coef_[0]})
selected_coefficients.to_csv(OUT / 'selected_model_coefficients.csv', index=False)

# %% 10. Independent validation performance, ROC and precision-recall curves
def evaluate(model, features, target):
    probability = model.predict_proba(features)[:, 1]
    predicted = (probability >= .5).astype(int)
    metrics = {'accuracy': float(accuracy_score(target, predicted)),
        'precision': float(precision_score(target, predicted, zero_division=0)),
        'recall': float(recall_score(target, predicted)),
        'f1': float(f1_score(target, predicted)),
        'roc_auc': float(roc_auc_score(target, probability)),
        'average_precision': float(average_precision_score(target, probability)),
        'log_loss': float(log_loss(target, probability))}
    return metrics, probability, predicted
validation, probability, predicted = evaluate(selected, X_val, y_val)
base_validation, base_probability, _ = evaluate(baseline, X_val, y_val)
dummy = DummyClassifier(strategy='prior').fit(X_train, y_train)
dummy_metrics, _, _ = evaluate(dummy, X_val, y_val)
cm = confusion_matrix(y_val, predicted, labels=[0, 1])
validation_table = pd.DataFrame({'selected_'+best_name: validation,
    'baseline': base_validation, 'prior_dummy': dummy_metrics}).T
validation_table.to_csv(OUT / 'validation_metrics.csv')
pd.DataFrame({'cust_id': df.loc[X_val.index, 'cust_id'], 'actual_churn': y_val,
    'probability': probability, 'predicted_churn': predicted}).to_csv(OUT / 'validation_predictions.csv', index=False)
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for name, prob, color in [('Selected: '+best_name, probability, GREEN),
                         ('Baseline', base_probability, SECOND)]:
    fpr, tpr, _ = roc_curve(y_val, prob)
    precision, recall, _ = precision_recall_curve(y_val, prob)
    axes[0].plot(fpr, tpr, color=color, label=f'{name} (AUC {roc_auc_score(y_val, prob):.3f})')
    axes[1].plot(recall, precision, color=color,
        label=f'{name} (AP {average_precision_score(y_val, prob):.3f})')
axes[0].plot([0, 1], [0, 1], '--', color='#999999', label='Random ranking')
axes[1].axhline(y_val.mean(), ls='--', color='#999999', label='Churn prevalence')
axes[0].set(xlabel='False positive rate', ylabel='True positive rate', title='Validation ROC', xlim=(0, 1), ylim=(0, 1.02))
axes[1].set(xlabel='Recall', ylabel='Precision', title='Validation precision-recall', xlim=(0, 1), ylim=(0, 1.02))
for ax in axes:
    ax.legend(fontsize=8, loc='lower right' if ax == axes[0] else 'lower left')
fig.tight_layout()
fig.savefig(OUT / 'validation_curves.png', dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(4.7, 4))
from matplotlib.colors import LinearSegmentedColormap
ax.imshow(cm, cmap=LinearSegmentedColormap.from_list('roc_green', ['#F3F6F5', GREEN]))
for (i, j), count in np.ndenumerate(cm):
    ax.text(j, i, f'{count:,}', ha='center', va='center', fontsize=15,
        color='white' if count > cm.max()/2 else '#222222')
ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['0: retained', '1: churn'],
    yticklabels=['0: retained', '1: churn'], xlabel='Predicted', ylabel='Actual',
    title='Validation confusion matrix\nThreshold 0.5')
fig.tight_layout()
fig.savefig(OUT / 'confusion_matrix.png', dpi=180)
plt.close(fig)
rng = np.random.default_rng(SEED)
auc_bootstrap = []
actual = y_val.to_numpy()
for _ in range(300):
    indices = rng.integers(0, len(actual), len(actual))
    if len(np.unique(actual[indices])) == 2:
        auc_bootstrap.append(roc_auc_score(actual[indices], probability[indices]))
auc_ci = np.quantile(auc_bootstrap, [.025, .975]).tolist()
print(validation_table.round(4).to_string())
print('Confusion matrix [[TN, FP], [FN, TP]]:', cm.tolist(), '| ROC AUC 95% bootstrap CI:', auc_ci)
joblib.dump(selected, OUT / 'churn_model.joblib')

# %% 11. Assess whether transformation improved performance
best_cv = float(cv_summary.loc[best_name, ('roc_auc', 'mean')])
base_cv = float(cv_summary.loc['baseline', ('roc_auc', 'mean')])
paired = cv_detail.pivot(index='fold', columns='model', values='roc_auc')
delta = paired[best_name] - paired['baseline']
print('Selected-minus-baseline CV AUC:', best_cv-base_cv,
    '| Paired fold differences:', delta.tolist())

# %% 12. Brief report, provenance and reproducibility
summary = {'source': SOURCE.name, 'source_sha256': hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'sheet': excel.sheet_names[0], 'rows': len(df), 'columns': len(df.columns),
    'missing_cells': int(df.isna().sum().sum()), 'rows_with_missing': int(df.isna().any(axis=1).sum()),
    'class_counts': {str(k): int(v) for k, v in class_counts.items()},
    'training_n': len(y_train), 'validation_n': len(y_val), 'seed': SEED,
    't_test': ttest, 'selected_model': best_name, 'cv_roc_auc': best_cv,
    'baseline_cv_roc_auc': base_cv, 'cv_auc_improvement': best_cv-base_cv,
    'validation_metrics': validation, 'validation_auc_bootstrap_ci95': auc_ci,
    'confusion_matrix': cm.tolist(), 'categorical_references': references,
    'numeric_training_medians': {c: float(v) for c, v in zip(NUM, num_imputer.statistics_)},
    'categorical_training_modes': {c: str(v) for c, v in zip(CAT, cat_imputer.statistics_)},
    'versions': {'python': platform.python_version(), 'numpy': np.__version__,
        'pandas': pd.__version__, 'scipy': scipy.__version__, 'scikit-learn': sklearn.__version__,
        'matplotlib': matplotlib.__version__}}
(OUT / 'results.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
def table_md(frame):
    headings = ['Index'] + list(frame.columns)
    rows = ['| ' + ' | '.join(map(str, headings)) + ' |', '| ' + ' | '.join(['---']*len(headings)) + ' |']
    for index, row in frame.iterrows():
        rows.append('| ' + ' | '.join([str(index)] + [f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row]) + ' |')
    return '\n'.join(rows)
cv_display = pd.DataFrame({'ROC AUC mean': cv_summary[('roc_auc', 'mean')],
    'ROC AUC SD': cv_summary[('roc_auc', 'std')], 'AP mean': cv_summary[('average_precision', 'mean')],
    'F1 mean': cv_summary[('f1', 'mean')]})
report = f'''# Customer churn: logistic regression analysis

## Findings
The model selected using training-only cross-validation was **{best_name}**. On the 8,000-customer validation set, ROC AUC was **{validation['roc_auc']:.4f}** (300-resample percentile bootstrap 95% CI {auc_ci[0]:.4f}–{auc_ci[1]:.4f}), precision **{validation['precision']:.4f}**, recall **{validation['recall']:.4f}**, and F1 **{validation['f1']:.4f}** at probability threshold 0.5. Average precision (AP) was **{validation['average_precision']:.4f}**, compared with validation churn prevalence {y_val.mean():.4f}. AP summarizes the precision-recall curve and is distinct from ROC AUC.

## Data completeness and exploration (tasks 1–4)
Source: `{SOURCE.name}`, sheet `{excel.sheet_names[0]}`. The data frame contains {len(df):,} rows and {len(df.columns)} columns, with {int(class_counts[1]):,} churned customers ({y.mean():.2%}). churn_flag is the screenshot's churn variable, with 1 treated as churn and 0 as retained. cust_id contains unique identifiers and was excluded from the model. Binary home_owner is categorical; num_cont, tot_call and num_out are numeric counts. The original spelling marrital is retained. There are {summary['missing_cells']} missing cells across {summary['rows_with_missing']} records; age has 43 missing values, tot_bill 33 and fortune 28. No target values are missing and no customer IDs are repeated.

Numeric predictors: {', '.join(NUM)}. Categorical predictors: {', '.join(CAT)}. Units and the observation window were not supplied; no currency or time unit is assumed.

The observed mean ages are {age_summary.loc[0, 'mean']:.2f} for retained customers and {age_summary.loc[1, 'mean']:.2f} for churned customers. Histograms use identical bins and densities to permit comparison despite different group sizes. Missing ages are excluded from these descriptive plots.

![Age histograms](age_histograms.png)

Welch's two-sided independent-samples t-test compares observed tot_bill values without imputation. H0: the two population means are equal. Retained customers: n={len(bill0):,}, mean={bill0.mean():.4f}; churned customers: n={len(bill1):,}, mean={bill1.mean():.4f}. Churn-minus-retained difference={diff:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}], t={t_result.statistic:.4f}, df={t_result.df:.2f}, {p_text}, Cohen's d={diff/pooled_sd:.4f}. {'Reject' if t_result.pvalue < .05 else 'Do not reject'} H0 at alpha=0.05. The difference is large relative to within-group variation. This is an unadjusted association, with customer independence assumed. Complete-case inference may be biased if missing bills are not missing at random; statistical significance alone does not establish practical importance or causation.

## Missing values and modeling (tasks 5–8)
Use training medians for numeric missing values and training modes for categorical missing values. The filled CSV uses these training-only statistics for both subsets; cross-validation independently refits preprocessing within each fold. Split 80%/20% with stratification and random seed 42: {len(y_train):,} training and {len(y_val):,} validation records. The target is never imputed.

The baseline uses one-hot encoding with one reference level dropped, standardized numeric variables, and L2-regularized logistic regression (C=1, lbfgs, maximum 2,000 iterations). Compare this with Yeo-Johnson numeric transformations and a quadratic expansion of standardized age and tot_bill (squares and interaction). All candidates use identical shuffled stratified 5-fold splits of the training set. Model choice is based solely on mean training CV ROC AUC; no tuning uses validation outcomes.

{table_md(cv_display)}

## Interpretation and performance (tasks 9–11)
Baseline coefficient interpretation is provided separately because its scale is directly interpretable. Each numeric odds ratio represents a one-standard-deviation increase in the predictor, holding the other encoded predictors fixed. Categorical reference levels: {json.dumps(references)}. Coefficients are regularized estimates; they are not hypothesis-test results. Correlation between total and component usage measures can make individual coefficients unstable, and associations are not causal effects.

{table_md(coefficients.head(8).set_index('feature')[['coefficient', 'odds_ratio']])}

For example, the baseline associates a one-SD increase in yahoo_air with 2.07 times the churn odds, and a one-SD increase in age with 0.53 times the odds, conditional on the other predictors. These are odds changes, not percentage-point changes in churn probability. In the selected quadratic model, age and tot_bill effects also depend on their squares and interaction, so neither has a single constant odds ratio across all customers. The selected-model coefficient CSV records these transformed terms; the saved pipeline applies the correct training scaling before prediction.

Validation performance below uses threshold 0.5 for class predictions. The prior dummy predicts the training prevalence as a constant probability and provides a reference. Baseline validation results are descriptive; the selected model was already fixed by CV.

{table_md(validation_table)}

Confusion matrix: TN={cm[0,0]:,}, FP={cm[0,1]:,}, FN={cm[1,0]:,}, TP={cm[1,1]:,}. Recall describes the share of churned customers detected; precision describes the share of flagged customers who actually churned. Accuracy should be compared with the dummy because the class distribution may be uneven.

![ROC and precision-recall](validation_curves.png)

![Confusion matrix](confusion_matrix.png)

The selected candidate changes mean CV ROC AUC by {best_cv-base_cv:+.5f} versus the baseline. Paired fold AUC differences: {', '.join(f'{v:+.5f}' for v in delta)}. {'The baseline remained the best candidate; the tested transformations did not improve mean CV ROC AUC.' if best_name == 'baseline' else 'The transformation improved mean training CV ROC AUC in this experiment; five correlated folds do not establish that the gain will persist on future data.'} Validation AUC changes by {validation['roc_auc']-base_validation['roc_auc']:+.5f} relative to the baseline. No decision threshold was optimized on validation data.

## Conclusion and reproducibility (task 12)
This is a mock-data exercise. Generalization to real customers, future time periods, or other acquisition channels has not been established. The held-out validation set provides the requested out-of-sample performance check; it is not an additional third test set. Production evaluation would require a future-time sample and a threshold chosen from training data according to retention costs. The AUC bootstrap interval captures validation sampling variability conditional on the fitted model, not retraining or model-selection uncertainty.

Run `python analysis.py` from the project directory after installing `requirements.txt`. The accompanying notebook contains executed analysis cells, plots and this report. The original workbook is preserved. Detailed CSVs contain data quality, fold scores, split assignments, imputed data, coefficients and validation predictions. results.json records the source SHA-256, seed, fitted imputation values and dependency versions. ROC AUC uses probabilities; classification metrics use churn=1 as the positive class.
'''
(OUT / 'report.md').write_text(report, encoding='utf-8')
# A self-contained HTML report with embedded images, without external assets.
import html
parts = []
for block in report.split('\n\n'):
    if block.startswith('!['):
        filename = block.split('](')[1].rstrip(')')
        encoded = base64.b64encode((OUT / filename).read_bytes()).decode()
        parts.append(f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(filename)}">')
    elif block.startswith('|'):
        lines = block.splitlines()
        cells = lambda line: [v.strip() for v in line.strip('|').split('|')]
        parts.append('<table><thead><tr>' + ''.join('<th>'+html.escape(v)+'</th>' for v in cells(lines[0])) + '</tr></thead><tbody>' +
            ''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in cells(line))+'</tr>' for line in lines[2:]) + '</tbody></table>')
    elif block.startswith('#'):
        level = len(block) - len(block.lstrip('#'))
        parts.append(f'<h{level}>'+html.escape(block.lstrip('# '))+f'</h{level}>')
    else:
        parts.append('<p>'+html.escape(block).replace('**', '').replace('`', '')+'</p>')
css = 'body{font:16px/1.65 Segoe UI,Microsoft YaHei,Arial,sans-serif;color:#303536;max-width:1050px;margin:40px auto;padding:0 28px;background:white}h1,h2{color:#254C45}h1{border-bottom:3px solid #254C45;padding-bottom:14px}h2{font-size:22px;margin-top:32px}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th{background:#254C45;color:white}td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #ddd}tr:nth-child(even){background:#f3f6f5}img{max-width:100%;display:block;margin:20px auto}@media print{body{margin:0}h2{break-after:avoid}table,img{break-inside:avoid}}'
(OUT / 'report.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Customer churn analysis</title><style>'+css+'</style><body>'+'\n'.join(parts)+'</body></html>', encoding='utf-8')
print('Report saved:', OUT / 'report.html')
zh_intro = f'''<h1>客户流失逻辑回归分析</h1>
<h2>主要结论</h2><p>完成图片中的 12 项作业。40,000 条记录中，20,522 条为流失客户（51.31%）；共 104 个缺失单元格，分布在 {summary['rows_with_missing']} 条记录中。流失客户平均年龄 39.80，未流失客户 49.51；平均 tot_bill 分别为 146.51 和 102.31，差值 44.20，95% 置信区间为 43.64–44.75。Welch 双侧 t 检验 p&lt;0.001，Cohen's d=1.58；程序中的 p=0 是数值下溢，不能解释为概率严格为零。数据未说明账单的货币单位或观察期。</p>
<h2>方法与验证</h2><p>churn_flag 对应作业里的 churn，1 表示流失。排除唯一客户编号 cust_id。按目标分层划分 32,000 条训练记录和 8,000 条验证记录，随机种子 42。数字变量用训练中位数填补，类别变量用训练众数填补；每个交叉验证折内重新拟合填补、编码与变换，避免泄漏。home_owner 按类别处理，其他计数变量按数值处理。基础模型使用标准化、独热编码和 L2 正则化逻辑回归。</p>
<p>仅在训练集上进行 5 折交叉验证，比较基础模型、Yeo-Johnson 变换和年龄/账单二次项与交互项。根据平均 CV ROC AUC 选择 {best_name}，验证集不参与模型或阈值选择。三个方案的 CV AUC 分别为 {base_cv:.4f}、{cv_summary.loc['yeo_johnson', ('roc_auc','mean')]:.4f}、{cv_summary.loc['age_bill_quadratic', ('roc_auc','mean')]:.4f}。</p>
<h2>验证集结果</h2><p>ROC AUC={validation['roc_auc']:.4f}，95% bootstrap 区间 [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]；precision={validation['precision']:.4f}，recall={validation['recall']:.4f}，F1={validation['f1']:.4f}，accuracy={validation['accuracy']:.4f}，AP={validation['average_precision']:.4f}。正类为流失，分类阈值为 0.5。混淆矩阵：TN={cm[0,0]}、FP={cm[0,1]}、FN={cm[1,0]}、TP={cm[1,1]}。AP 是 PR 曲线的平均精确率指标，与 ROC AUC 不同。</p>
<p>二次变换的 CV AUC 比基础模型高 {best_cv-base_cv:.5f}，验证 AUC 高 {validation['roc_auc']-base_validation['roc_auc']:.5f}；但验证 F1 为 {validation['f1']:.4f}，基础模型为 {base_validation['f1']:.4f}。因此，变换对概率排序略有改善，并未提高默认阈值下的所有分类指标。</p>
<h2>结果解释与局限</h2><p>基础模型中，控制其他变量后，yahoo_air 每增加一个训练标准差对应的流失 odds 约为原来的 2.07 倍，age 每增加一个标准差约为 0.53 倍。odds 变化不等于流失概率的百分点变化。所选二次模型的年龄与账单效应取决于客户取值及交互项，不能用单一恒定 odds ratio 概括。总量和组成变量可能相关，系数不能解释为因果效应或显著性检验。</p>
<p>这是模拟数据。当前结果来自一次独立的验证划分，没有额外第三个测试集，也没有证明真实客户或未来时段的表现。若要用于业务，应另外做时间外验证，并根据留存成本在训练数据中选择阈值。下面附完整英文报告、各折比较、图表及方法细节。</p>'''
(OUT / 'report_zh.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>客户流失分析</title><style>'+css+'</style><body>'+zh_intro+'\n'.join(parts)+'</body></html>', encoding='utf-8')
