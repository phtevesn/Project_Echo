# Project Echo: Socio-Economic Income Predictor

> AI-based **regression** model that predicts an individual's estimated annual income
> using societal factors from the **ACS PUMS (American Community Survey)**
> dataset, accessed via `fairlearn.datasets.fetch_acs_income`.

_Last updated: May 2026_

---

## What Is This?

Project Echo is a machine-learning application that predicts financial worthiness
(estimated annual personal income in USD) based on socio-economic, socio-cultural,
and demographic features from the ACS PUMS Income dataset.

The model is an **ensemble of XGBoost, LightGBM, and CatBoost regressors**,
trained on log-transformed income (`log1p(PINCP)`) and averaged at inference time.
Predictions are converted back to dollar scale via `exp()`.

A **Python + tkinter GUI** (`app.py`) lets any user enter their personal information
and instantly receive a predicted income estimate from the saved pipeline
`income_pipeline.pkl` (loaded via `joblib`).

---

## Files

| File | Description |
|------|-------------|
| `app.py` | Main Python tkinter GUI application |
| `income_pipeline.pkl` | Pre-trained ensemble model (**must be present**) |
| `README.md` | This file |

---

## Requirements

Install all dependencies with one command:

```
pip install scikit-learn joblib pandas numpy xgboost lightgbm catboost fairlearn
```

| Package | Purpose |
|---------|----------|
| `scikit-learn` | Preprocessing (OrdinalEncoder, QuantileTransformer, etc.) |
| `joblib` | Load `income_pipeline.pkl` |
| `pandas` | Build input DataFrames |
| `numpy` | Numerical operations |
| `xgboost` | XGBoost regressor (in ensemble) |
| `lightgbm` | LightGBM regressor (in ensemble) |
| `catboost` | CatBoost regressor (in ensemble) |
| `fairlearn` | Source of the ACS PUMS Income dataset (training only) |

> `tkinter` is included with the standard Python installer on Windows. On Ubuntu/Debian run `sudo apt-get install python3-tk` if missing.

---

## Option A – Run from Python Source (Windows / Mac / Linux)

### Prerequisites

- **Python 3.9 or later** → <https://www.python.org/downloads/>
  _(tick "Add Python to PATH" during installation on Windows)_
- Install required packages (see [Requirements](#requirements) above):
  ```
  pip install scikit-learn joblib pandas numpy xgboost lightgbm catboost fairlearn
  ```

### Steps

1. Copy `income_pipeline.pkl` into the same folder as `app.py`.
2. Open a terminal / Command Prompt in that folder:
   - **Windows:** Shift + right-click the folder → _"Open PowerShell window here"_
     or press `Win+R`, type `cmd`, then `cd "path\to\folder"`
3. Run:
   ```
   python app.py
   ```
4. The Project Echo window opens. Fill in the form and click **Predict Estimated Income**.

---

## Option B – Compile a Stand-Alone Windows .EXE

> No Python installation needed on the target machine.

### Prerequisites _(build machine only)_

- Python 3.9+ with the packages from Option A
- PyInstaller:
  ```
  pip install pyinstaller
  ```

### Build Command

Run once from the project folder (same directory as `app.py`):

```bat
pyinstaller --onefile --windowed ^
  --add-data "income_pipeline.pkl;." ^
  --name "ProjectEcho" ^
  app.py
```

_On Mac/Linux replace the semicolon with a colon: `--add-data "income_pipeline.pkl:."`_

### Output

A single executable is created at `dist\ProjectEcho.exe`.  
Copy **only** `ProjectEcho.exe` to any Windows PC – double-click to launch.

---

## Option C – Run Over the Web (WWW / Hosted Server)

The tkinter GUI is a desktop application. To expose it over the web:

### Path 1 – Streamlit Web App _(recommended)_

```
pip install streamlit
streamlit run web_app.py
```

Then open <http://localhost:8501> in any browser.

**Free public deployment:**
- Push the project to GitHub.
- Go to <https://share.streamlit.io> and connect your repo.
- Streamlit Community Cloud hosts the app publicly at a shareable URL.

### Path 2 – Flask / FastAPI REST Backend

```
pip install flask
```

Create a small API endpoint that accepts JSON inputs, calls `pipeline.predict()`,
and returns the result. Host on any cloud provider (Render, Railway, Heroku, etc.)
and pair with any HTML / React front-end.

---

## Input Fields Reference (ACS PUMS Income Dataset)

| Field (ACS code) | Type | Description / Options |
|---|---|---|
| Age (`AGEP`) | Integer | Age in years (16–95) |
| Class of Worker (`COW`) | Dropdown | Employment sector: Private for-profit, Private non-profit, Local/State/Federal gov, Self-employed (inc. / not inc.), Unpaid, Unemployed |
| Education (`SCHL`) | Dropdown | Grouped: Less than HS · HS/GED · Some College · Associate's · Bachelor's · Graduate+ |
| Hours Worked / Week (`WKHP`) | Integer | Usual hours worked per week (1–99) |
| Sex (`SEX`) | Dropdown | Male / Female |
| Occupation (`OCCP`) | Dropdown | 26 ACS occupation groups (MGR, ENG, MED, SAL, TRN, etc.) |
| Marital Status (`MAR`) | Dropdown | Married · Widowed · Divorced · Separated · Never married |
| Household Role (`RELP`) | Dropdown | Reference person, Spouse, Child/Dependent, Parent, Extended family, Non-family, Other, Group quarters |
| Race (`RAC1P`) | Dropdown | White · Black · Asian · Indigenous · Pacific Islander · Multi-racial · Other |
| Place of Birth (`POBP`) | Dropdown | US state (FIPS code) or foreign-born region group |

**Output:** Estimated annual personal income in USD (e.g. `$72,400`), derived by
averaging the log-income predictions from XGBoost, LightGBM, and CatBoost, then
applying `exp()` to convert back to dollar scale.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `income_pipeline.pkl not found` | Place the `.pkl` file in the same directory as `app.py` (or the `.exe`) and restart. |
| `ModuleNotFoundError: joblib` | Run `pip install joblib` |
| `ModuleNotFoundError: xgboost / lightgbm / catboost` | Run `pip install xgboost lightgbm catboost` |
| `No module named 'tkinter'` | Reinstall Python from python.org and tick "tcl/tk and IDLE", or on Linux: `sudo apt-get install python3-tk` |
| Prediction seems off | Ensure the `.pkl` was generated by `final_income_model.ipynb` using `fairlearn.datasets.fetch_acs_income` (ACS PUMS dataset). |

---

## Contact / Project

- **Repository:** <https://github.com/Zon-Vorelle/Project_Echo>
- **Project:** Project Echo: Socio-Economic Income Predictor
