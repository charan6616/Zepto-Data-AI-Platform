# /analytics — Titanic: analyst-to-data-scientist workflow (one pass)

This module turns the classic **Titanic** dataset (Seaborn built-in loader) into a full analyst pipeline in
one cohesive pass: **profile → clean defensibly → tell a visual data story → build and rigorously evaluate a
predictive-modeling pipeline** (classification + a regression side-task) on the *same* data.

**Single-load rule.** The raw dataset is fetched exactly **once**, in `01_eda.ipynb` via
`sns.load_dataset('titanic')` (network on first run, then Seaborn's local cache). It is immediately
committed as the module's offline fallback, `titanic.csv`, via `df.to_csv("titanic.csv", index=False)`.
`02_modeling.ipynb` continues straight from that committed CSV with `pd.read_csv("titanic.csv")` — the
dataset is **never loaded a second time** anywhere in the module, so everything can be re-graded offline.

---

## 1 · Part A — Profiling, cleaning & the data story (`01_eda.ipynb`)

### 1.1 Profile (Task 1)
- `df.shape` → **(891, 15)**
- `df.info()` and `df.describe()` are printed in the notebook (columns: `survived, pclass, sex, age, sibsp,
  parch, fare, embarked, class, who, adult_male, deck, embark_town, alive, alone`).
- Raw DataFrame committed immediately → `titanic.csv`.

### 1.2 Missing values & strategy (Task 2)

Exact percentages measured on the raw load, with the threshold-rule decision per column:

| column        | missing % | count | rule band                   | strategy                                 |
|---------------|-----------|-------|-----------------------------|------------------------------------------|
| `deck`        | 77.2166   | 688   | &gt; 30% (imputation unreliable) | **drop the column**                 |
| `age`         | 19.8653   | 177   | 5%–30%                      | impute with median                        |
| `embarked`    | 0.2245    | 2     | under 5%                    | drop the rows                             |
| `embark_town` | 0.2245    | 2     | under 5%                    | drop the rows (same 2 rows)               |

**Rule:** missing **< 5%** → drop rows; **5%–30%** → impute; **> 30%** → drop the column or encode "missing"
as its own category.

**Why `deck` is dropped (not imputed, not re-coded):** at **77.22%** missing, imputation would fabricate
~77% of the column — guessing far more data than we measure — and would inject fictitious signal into every
downstream statistic. The standard alternative (encode "missing" as its own category) would create a
near-constant 8th bucket holding ~77% of passengers, which carries almost no discriminative information
(deck letter already correlates strongly with class/fare, which stay in the data). **Decision: drop `deck`.**
`age` (19.87%) is median-imputed (median = 28.0) per the 5–30% band; the 2 `embarked`/`embark_town` rows
(0.22%) are dropped per the under-5% rule. Cleaned frame: **(889, 14), zero missing values.**

### 1.3 Univariate (Task 3)
- Histograms and box plots for `age` and `fare` (saved under `charts/`).
- **IQR outliers** (values outside `[Q1 − 1.5·IQR, Q3 + 1.5·IQR]`): `age` → **65** of 889 (Q1 22, Q3 35,
  IQR 13); `fare` → **114** of 889 (Q1 7.90, Q3 31, IQR 23.10).
- `fare` central tendency: **mean = 32.10, median = 14.45, mode = 8.05**.
- **Skewness verdict:** because **mean > median > mode**, `fare` is **right-skewed (positively skewed)** — a
  long right tail of expensive first-class tickets pulls the mean far above the typical (median) fare while
  most tickets pile up at the low, most-frequent fare of $8.05.

### 1.4 Bivariate (Task 4)
Survival rates via **boolean masking with `&`/`|`** (e.g. `df.loc[(df.sex == 'female') & (df.pclass == 1), 'survived'].mean()`):

| breakdown        | survival rate |
|------------------|---------------|
| sex = female     | **74.04%**    |
| sex = male       | **18.89%**    |
| pclass = 1       | **62.62%**    |
| pclass = 2       | **47.28%**    |
| pclass = 3       | **24.24%**    |
| female, pclass 1 | **96.74%**    |
| female, pclass 2 | **92.11%**    |
| female, pclass 3 | **50.00%**    |
| male, pclass 1   | **36.89%**    |
| male, pclass 2   | **15.74%**    |
| male, pclass 3   | **13.54%**    |

**Correlation matrix** (`charts/corr_heatmap.png`) is restricted to **exactly** the six numeric columns
`survived, pclass, age, sibsp, parch, fare`. The boolean flags `adult_male` and `alone` are **excluded** —
they are derived/redundant (computable from sex/age and from sibsp+parch), not independent measured
features.

**The two strongest correlations** (largest |r| among all off-diagonal pairs):
1. **`pclass` ↔ `fare`, r = −0.5482** — the strongest pair, and the anchor of the whole story. First class
   paid the highest fares and 3rd the lowest; since class drives survival negatively (r = −0.3355) while
   fare drives it positively (r = 0.2553), this pair captures the full socio-economic axis: ticket price =
   class = survival odds.
2. **`sibsp` ↔ `parch`, r = 0.4145** — the two family-size components move together (people travelling with
   siblings/spouses also tended to travel with parents/children). Together they define the "travelling with
   family" dimension — the exact redundancy that makes `alone` derivable. Their direct links to survival are
   weak (−0.034 and 0.083).

### 1.5 Multivariate data story (Task 5) — six charts, each interpreted
1. **Survival by sex** (`bar_surv_by_sex.png`)… women **74.0%** vs men **18.9%** (≈4× gap). *"Women and
   children first"* is the mechanism; being female was worth ~55 points of survival probability.
2. **Survival by class** (`bar_surv_by_pclass.png`)… **62.6% → 47.3% → 24.2%**, a monotonic socio-economic
   gradient. Class encodes wealth, cabin position and lifeboat proximity; odds drop at every rung.
3. **Sex × class** (`bar_surv_sex_pclass.png`)… female/1st **96.7%** and female/2nd **92.1%** vs male/2nd
   **15.7%** and male/3rd **13.5%** — an ~83-point spread proving survival was a structured privilege
   gradient, not a coin flip.
4. **Fare by class & survival** (`box_fare_pclass_survived.png`)… survivors' fare distributions sit above
   non-survivors' inside every class, and fares soar from 3rd to 1st. This is the visual form of
   `pclass ↔ fare` (r = −0.55): money bought cabin position, and cabin position bought a lifeboat place.
5. **Age vs fare by survival** (`scatter_age_fare.png`)… survivors spread across medium-to-high fares at
   every age; the dense low-fare 3rd-class cloud is dominated by non-survivors. Age shows no blanket rule
   once class/fare are known (r = −0.07 — the weakest of the six).
6. **Pair plot** (`pairplot.png`)… survivors sit at higher fare, low pclass, and family groups (r = 0.41);
   age panels overlap heavily. **Argument:** *female + 1st class + higher fare → survive; male + 3rd class +
   low fare → perish*, with family size as a modest modifier.

### 1.6 Exploratory standardization check (Task 6, EDA-only)
`z = (x − mean) / std` applied to `age` and `fare` on the cleaned frame (`charts/zscore_before_after.png`):

| column | raw mean | raw std | z mean | z std |
|--------|----------|---------|--------|-------|
| age    | 29.3152  | 12.9849 | ≈ 0 (2.8e−16) | 1.0000 |
| fare   | 32.0967  | 49.6975 | ≈ 0 (1.4e−16) | 1.0000 |

Both transformed columns have **mean ≈ 0, std = 1** (residuals are ~1e−16 float noise). Purely a sanity
check — the modeling pipeline performs its own train-only scaling.

---

## 2 · Part B — Predictive modeling (`02_modeling.ipynb`)

Continues from the committed `titanic.csv`. **Features** used by the classifier pipeline:
`pclass, sex, age, sibsp, parch, fare, embarked`; target `survived`.

### 2.1 Stratified split (Task 7)
Class balance: **38.4% survived / 61.6% not** (moderately imbalanced). `stratify=y` forces both folds to
mirror the population proportions (train **0.3834**, test **0.3855**) so random-split drift can't bias the
metrics; the split happens **before any preprocessing**, so no test information can leak into fitted steps.
`test_size=0.20, random_state=42`. Train 712 rows, test 179 rows.

### 2.2 Preprocessing (Task 8)
`ColumnTransformer` wrapped in a `Pipeline`, fitted **only on `X_train`**:
- numeric (`pclass, age, sibsp, parch, fare`) → `SimpleImputer(median)` + `StandardScaler`;
- categorical (`sex, embarked`) → `SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first',
  handle_unknown='ignore')`.

The test split is passed through the same fitted transformers in transform-only mode. Choice statement:
age's ~19.9% NaN is median-imputed *inside* the pipeline (a deliberate, restated choice for part B), and the
modest class imbalance is treated separately in Task 10.

### 2.3 The three classifiers (Task 9) — identical split

| model             | train acc | accuracy | precision | recall | F1    | AUC   |
|-------------------|-----------|----------|-----------|--------|-------|-------|
| LogisticRegression| 0.8076    | 0.8045   | 0.7931    | 0.6667 | 0.7244| 0.8435|
| DecisionTree      | 0.9831    | 0.8212   | 0.7937    | 0.7246 | 0.7576| 0.7914|
| RandomForest (200)| 0.9831    | 0.8101   | 0.7778    | 0.7101 | 0.7424| 0.8310|

Also reported: confusion-matrix heatmaps (`charts/confusion_matrices.png`), an overlaid ROC plot with per-model
AUC (`charts/roc_curves.png`), and the fitted decision tree rendered via `plot_tree` with feature/class
labels (`charts/decision_tree.png`).

### 2.4 Imbalance handling (Task 10) — LogisticRegression, 3 ways

| variant                  | precision | recall | F1    | accuracy |
|--------------------------|-----------|--------|-------|----------|
| baseline (no handling)   | 0.7931    | 0.6667 | 0.7244| 0.8045   |
| `class_weight='balanced'`| 0.7297    | 0.7826 | 0.7552| 0.8045   |
| **SMOTE (train-only)**   | **0.7397**| **0.7826** | **0.7606** | **0.8101** |

**Conclusion:** the baseline's default 0.5 threshold under-predicts survivors (high precision, worst recall
0.667). Both remedies raise recall to **0.783**. **SMOTE wins** with the best F1 (**0.7606**) at equal-or-
better accuracy (0.810), because it re-balances by *re-sampling the minority class's actual feature
distribution* rather than only re-weighting the loss — a better fit for the logistic boundary here. The gain
is modest (+0.036 F1) since the 38/62 imbalance is only moderate. SMOTE is applied inside the pipeline, i.e.
**to the training fold only** (test data never resampled, no leakage).

### 2.5 Hyperparameter tuning (Task 11)
`RandomForestClassifier(oob_score=True, ...)` (OOB requires `oob_score=True` at construction) tuned by
`GridSearchCV` (5-fold, `roc_auc`, `n_estimators ∈ {100,200,300}`, `max_depth ∈ {None,7,12}`,
`max_features ∈ {sqrt, log2, None}`):
- **best params:** `n_estimators=300, max_depth=7, max_features='sqrt'`
- best CV ROC-AUC = **0.8711**
- **OOB score (`oob_score_`) = 0.8287**
- tuned RF test AUC = **0.8468**, test F1 = 0.7107, test accuracy = 0.8045

### 2.6 Regression side-task (Task 12) — predict `fare` (multivariate linear regression)
- **MAE = 20.8094** · **RMSE = 30.4731** · **R² = 0.3999** · **Adjusted R² = 0.3753** (n = 179, k = 7)
- Residual plot: `charts/residuals.png`.
- **Heteroscedasticity verdict: YES** — residuals fan out with fitted value: median |residual| is **10.44**
  on the low-predicted half vs **24.97** on the high-predicted half, and corr(predicted, |residual|) = **+0.55**.
  Expected, because the strongly right-skewed `fare` target (Task 3) makes cheap tickets easy to predict
  while expensive first-class fares carry huge variance. (Remedies — log-target or robust regression — are
  noted as out of scope.)

### 2.7 Final comparison tables & recommendation (Task 13)

**Group 1 — classifier metrics (test set):**

| metric          | LogisticRegression | DecisionTree | RandomForest |
|-----------------|--------------------|--------------|--------------|
| accuracy        | 0.8045             | 0.8212       | 0.8101       |
| precision       | 0.7931             | 0.7937       | 0.7778       |
| recall          | 0.6667             | 0.7246       | 0.7101       |
| F1              | 0.7244             | 0.7576       | 0.7424       |
| AUC             | 0.8435             | 0.7914       | 0.8310       |

**Group 2 — regression metrics (predict `fare`, test set):**

| metric             | value    |
|--------------------|----------|
| MAE                | 20.8094  |
| RMSE               | 30.4731  |
| R-squared          | 0.3999   |
| Adjusted R-squared | 0.3753   |

> Classification and regression metrics are on **different scales** and are **not directly comparable** —
> they are presented as two distinct metric groups, one per model type, never merged into a single
> comparable number set.

**Final recommendation (deploy the tuned Random Forest).** The tuned Random Forest
(`n_estimators=300, max_depth=7, max_features='sqrt'`) achieved the best cross-validated ROC-AUC of the
whole module (**0.8711**), the best test-set AUC (**0.8468**), and a strong out-of-bag score (**0.8287**)
that is close to its CV score — evidence it generalizes without overfitting. It also produced robust
confusion-matrix behavior across the full metric suite (test accuracy 0.8045, precision 0.778, recall 0.710,
F1 0.711 per its tuned config). Logistic Regression is the close runner-up (AUC 0.8435) and the better pick
if you need a small, interpretable, quickly servable model, while the plain Decision Tree's highest raw
accuracy (0.8212) is offset by the lowest AUC (0.7914) and tree-level variance. On balance, ranking quality
plus stability wins: **deploy the tuned Random Forest pipeline** (`pipeline_best.joblib`).

### 2.8 Deployment artifact (Task 14)
`pipeline_best.joblib` is the **complete fitted pipeline** — the `ColumnTransformer` (median imputer, one-hot
encoder, scaler) *together with* the tuned Random Forest — saved via `joblib.dump(...)`. The notebook then
reloads it with `joblib.load` and verifies: (a) predictions are **identical** to the pre-reload object on the
same raw rows, and (b) raw rows with deliberately **missing `age`** still predict correctly (imputation
happens inside the pipeline). No bare estimator was saved: the artifact scores raw, unpreprocessed data
end-to-end.

---

## 3 · Module layout

```
analytics/
├── 01_eda.ipynb          # Part A — single raw load, profile, clean, EDA story, z-score check
├── 02_modeling.ipynb     # Part B — read titanic.csv, models, imbalance, tuning, regression, deploy
├── titanic.csv           # committed offline fallback (raw 891×15, the one and only load snapshot)
├── pipeline_best.joblib  # complete fitted pipeline (preprocessing + tuned RandomForest)
├── charts/               # 14 chart PNGs produced by the notebooks
└── README.md             # this file
```

**Regeneration:** to rebuild from scratch, run the notebooks in order
(`01_eda.ipynb` first — it creates `titanic.csv` from the single `sns.load_dataset` call; then
`02_modeling.ipynb`, which reads `titanic.csv`). All artifacts (charts, CSV, joblib) regenerate in place.