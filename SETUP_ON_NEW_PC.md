# Setup on a New PC

You already have the full project (code, notebooks, dashboard, README) copied
over. The only thing missing is the data and the trained models, because those
are too big to email. This guide rebuilds all of it in a few minutes.

Everything uses a fixed random seed (`random_state=42`), so the regenerated
data, models, and metrics come out **identical** to the original PC.

---

## Step 0 - Prerequisites

- Install **Python 3.10 or newer** (3.13 is what the project was built on).
- Open a terminal (PowerShell or Command Prompt) and make sure Python works:

  ```
  python --version
  ```

---

## Step 1 - Go into the project folder

Navigate into the unzipped project folder. For example:

```
cd path\to\credit-card-fraud-detection
```

All the commands below are run from inside this folder.

---

## Step 2 - Install the dependencies

```
pip install -r requirements.txt
```

This installs pandas, scikit-learn, XGBoost, imbalanced-learn, SHAP, Streamlit,
matplotlib, seaborn, and Jupyter.

---

## Step 3 - Add the raw dataset

1. Download `creditcard.csv` from Kaggle:
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
2. Put the file here (the name must be exactly `creditcard.csv`):

   ```
   data/raw/creditcard.csv
   ```

If the `data/raw/` folder does not exist, create it first.

---

## Step 4 - Rebuild the data and models

Run these three commands in order. Each one recreates files that were left out
of the transfer:

```
python src/preprocess.py     # creates data/processed/train.csv, test.csv, models/scaler.pkl
python src/train.py          # trains the 3 models -> models/*.pkl + best_model.pkl
python src/evaluate.py        # creates report figures + reports/model_comparison.csv
```

- `preprocess.py` takes a few seconds.
- `train.py` takes a couple of minutes (it trains 3 models with cross-validation).
- `evaluate.py` takes about a minute (it also builds the SHAP plots).

---

## Step 5 (optional) - Run the notebooks

If you also want the notebooks to show their saved charts and outputs:

```
python -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/02_preprocessing.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/03_modeling.ipynb
python -m nbconvert --to notebook --execute --inplace notebooks/04_evaluation.ipynb
```

`01_eda.ipynb` is what generates `class_distribution.png` and
`correlation_heatmap.png` in `reports/figures/`.

---

## Step 6 - Launch the dashboard

```
python -m streamlit run dashboard/app.py
```

Your browser should open automatically. If it does not, copy the
`http://localhost:8501` (or `8502`) URL printed in the terminal into your
browser. Keep the terminal open while using the dashboard; press `Ctrl+C` to
stop it.

---

## Quick reference - the whole thing in one block

```
cd path\to\credit-card-fraud-detection
pip install -r requirements.txt
# (place creditcard.csv into data/raw/ from Kaggle)
python src/preprocess.py
python src/train.py
python src/evaluate.py
python -m streamlit run dashboard/app.py
```

---

## What gets regenerated (was NOT transferred)

| File / folder                | Recreated by      |
|------------------------------|-------------------|
| `data/raw/creditcard.csv`    | Kaggle download   |
| `data/processed/train.csv`   | `src/preprocess.py` |
| `data/processed/test.csv`    | `src/preprocess.py` |
| `models/scaler.pkl`          | `src/preprocess.py` |
| `models/*.pkl` (all models)  | `src/train.py`    |
| `reports/figures/*.png`      | `src/evaluate.py` + `notebooks/01_eda.ipynb` |
| `reports/model_comparison.csv` | `src/evaluate.py` |

---

## Troubleshooting

- **`streamlit` is not recognized** → use `python -m streamlit run dashboard/app.py`.
- **`FileNotFoundError` about creditcard.csv** → the file is missing or misnamed;
  it must be exactly `data/raw/creditcard.csv`.
- **`ModuleNotFoundError`** → re-run `pip install -r requirements.txt`.
- **Dashboard says "Processed data not found"** → you skipped Step 4; run
  `python src/preprocess.py` and `python src/train.py` first.
