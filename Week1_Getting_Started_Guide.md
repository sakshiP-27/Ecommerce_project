# Week 1 Guide: Purchase-Intent Prediction Project
### A beginner-friendly walkthrough — Foundations & Data

---

## Before You Start

**Goal for this week:** Get a basic (even bad) version of the whole pipeline working — raw data in, a prediction out. Don't aim for a good model yet. Aim for a *working* one.

**Why this matters:** Most student projects fail because pieces never get connected — a great notebook here, a half-built API there, nothing talking to each other. If you get one ugly-but-complete pipeline running this week, everything after this is just *improving* it, not building it from scratch.

**What you need before starting:**
- `ecommerce_sessions.csv` (the data)
- `data_dictionary.csv` (explains what each column means)
- Python 3.10+ installed, with a virtual environment (never install packages directly into your system Python)

---

## Step 1: Set Up Your Project Folder

Create this folder structure. It looks like a lot, but you're just making empty folders for now — you'll fill them in over the coming weeks.

```
purchase-intent-xai/
├── dataset/              ← put your 2 CSV files here
├── notebooks/            ← your exploration notebooks go here
├── src/
│   ├── config/           ← settings file
│   ├── data/              ← code that loads and checks the data
│   ├── preprocessing/     ← code that cleans/transforms data
│   ├── features/          ← code that creates new features
│   ├── models/             ← code that trains models
│   ├── explain/            ← code that explains predictions (later)
│   ├── api/                ← the web service (later)
│   └── utils/               ← small helper code (logging etc.)
├── tests/                ← automated tests
├── docker/               ← for later (deployment)
├── .github/workflows/    ← for later (automation)
├── .gitignore
├── .env.example
└── README.md
```

**How to do it:**
1. Create one main folder called `purchase-intent-xai`.
2. Inside it, create all the subfolders shown above (empty for now — that's fine).
3. Put your two CSV files inside `dataset/`.
4. Run `git init` inside the main folder to start tracking your work with Git.
5. Add a `.gitignore` file so you don't accidentally commit junk files (search "python gitignore template" if unsure what to put in it).

✅ **Done when:** You have the folder structure above, and `git status` works without errors.

---

## Step 2: Create One Settings File (Config)

**What this means:** Instead of typing the same file paths, numbers, and settings over and over in different notebooks (and forgetting to update one of them later), you put them all in **one file**. Every other piece of code reads from this file.

**Why it matters:** If you hard-code things like `pd.read_csv("../data/sessions.csv")` in five different notebooks, and then you move the file, you now have five things to fix. One config file = one thing to fix.

**What to include in your config:**
- Path to the dataset file
- A "random seed" number (e.g. `42`) — this makes your results repeatable every time you run the code
- What fraction of data to hold out for testing (e.g. `0.2` = 20%)
- The name of the target column (`Converted`)
- The list of columns that are categories, not numbers (this includes `OperatingSystems`, `Browser`, `Region`, `TrafficType` — more on this below)
- The prediction threshold (you'll set this properly later — `0.5` is a fine placeholder for now)

**How to do it:** Create a file `src/config/config.yaml` (a simple text file) with these values written as `key: value` pairs. Example:

```yaml
data_path: "dataset/ecommerce_sessions.csv"
random_seed: 42
test_size: 0.2
target_column: "Converted"
categorical_columns:
  - "OperatingSystems"
  - "Browser"
  - "Region"
  - "TrafficType"
  - "VisitorType"
  - "Month"
  - "Weekend"
threshold: 0.5
```

✅ **Done when:** You have one `config.yaml` file, and it's the only place these values are written down.

---

## Step 3: Set Up Logging (Instead of `print()`)

**What this means:** Python has a built-in `logging` module that's a more professional version of `print()`. It timestamps messages, labels their severity (info, warning, error), and can be turned on/off easily.

**Why it matters:** A project full of `print("here")` and `print("got the data")` looks unfinished. Logging is a small change that makes your code look — and behave — like real production code.

**How to do it:** In `src/utils/`, create a small helper file that sets up logging once, so you can reuse it everywhere:

```python
import logging

def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    return logging.getLogger(name)
```

Then in any other file, instead of `print("loading data")`, you'd write:

```python
from src.utils.logger import get_logger
logger = get_logger(__name__)
logger.info("Loading data...")
```

✅ **Done when:** No `print()` statements are left in your actual source code (`src/`) — only in throwaway notebook cells if needed.

---

## Step 4: Write the Data Loader

**What this means:** A small piece of code whose only job is: open the CSV, check it looks correct, and hand back a clean pandas DataFrame. If something's wrong (file missing, columns don't match), it should fail with a clear error message — not silently continue with broken data.

**Why it matters:** Later, your EDA notebook, your training code, your API, and your dashboard will *all* load the data. If they each do it slightly differently, you'll get subtle bugs. One loader function used everywhere avoids this.

**What it should do:**
1. Read the CSV path from your config file (not hard-coded).
2. Check that all expected columns are present (compare against `data_dictionary.csv`).
3. Raise a clear error if the file is missing or a column is missing.
4. Log a message when loading succeeds (e.g. "Loaded 12000 rows, 18 columns").

**Simple starting point:**

```python
import pandas as pd
from pathlib import Path

def load_sessions(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    df = pd.read_csv(path)

    expected_columns = ["Administrative", "Informational", "ProductRelated",
                         "BounceRates", "ExitRates", "PageValues",
                         "Month", "VisitorType", "Weekend", "Converted"]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    return df
```

(You'll expand the `expected_columns` list to the full set once you check the data dictionary.)

✅ **Done when:** Calling this function twice gives identical results, and it raises a clear error if you rename or delete a column to test it.

---

## Step 5: Build the "Thin Slice" — Your First (Bad) Prediction

**What this means:** Get one prediction working, start to finish, even with the simplest possible model. This is the single most important task of Week 1.

**The chain you're building:**
```
CSV file → data loader → very basic model → save the model → load it back → predict on one row
```

**How to do it, step by step:**
1. Load the data using your loader from Step 4.
2. Split off the target column (`Converted`) from the rest.
3. For now, just drop or roughly encode any non-numeric columns — don't worry about doing this "properly" yet (that's Step 8).
4. Split into train/test using `train_test_split` (use your `random_seed` from config!).
5. Train the simplest model you can: `LogisticRegression()` from scikit-learn.
6. Check it runs and gives *some* accuracy — it doesn't need to be good.
7. Save the trained model using `joblib.dump(model, "models/model_v0.pkl")`.
8. Write a tiny script that loads the model back and predicts on one example row.

**This is intentionally rough.** You are *not* trying to build the real model this week — you're proving the whole chain works so nothing surprises you later.

✅ **Done when:** You can run one script and see a predicted probability printed for a sample session.

---

## Step 6: Explore the Data (EDA)

**What this means:** Before building the real model, get to know the data — what a "typical" buying session looks like versus a non-buying one, and whether anything looks suspicious.

**Do this in a notebook** (`notebooks/01_eda.ipynb`), and for every chart, write **one sentence below it** stating what it tells you. A reviewer skimming your notebook should be able to understand your findings from those one-liners alone.

**Things to check:**
1. **Overall conversion rate** — what % of sessions ended in a purchase? (Should be around 15–16%.)
2. **Buyers vs. non-buyers** — do buyers spend more time on product pages? Have higher `PageValues`? Lower `ExitRates`?
3. **Seasonality** — does conversion rate change by `Month` or around `SpecialDay`?
4. **Returning vs. new visitors** — does `VisitorType` affect conversion?
5. **Correlations** — which numeric columns move together?
6. Make **at least one interactive chart** using Plotly (not just static matplotlib/seaborn images).

⚠️ **Two important traps to watch for while exploring:**

- **`OperatingSystems`, `Browser`, `Region`, `TrafficType` are stored as numbers, but they are NOT real numbers** — they're category codes. "Region 3" isn't more than "Region 1," the same way "zip code 90210" isn't more than "zip code 10001." Treat these as categories, not quantities.
- **`PageValues` is a very strong predictor — almost suspiciously strong.** It reflects value from pages seen close to checkout. Ask yourself: if you were scoring this session in real time, early in the visit, would you actually know this value yet? Write down your answer — you'll need to justify this decision later.

✅ **Done when:** Your EDA notebook runs top to bottom without errors, and every chart has a one-line takeaway underneath it.

---

## Step 7: Build the Real Preprocessing Pipeline

**What this means:** Now that you understand the data, build one clean, reusable pipeline that transforms raw data into model-ready data — properly this time (unlike the rough version in Step 5).

**What it needs to do:**
1. **Encode categorical columns** (turn categories into numbers the model can use) — this includes `VisitorType`, `Month`, `Weekend`, AND the four "sneaky" ID columns from Step 6.
2. **Scale/transform the numeric duration columns** — these are heavily skewed (lots of zeros, a few huge values), so a plain "just scale it" approach won't work well. Look up `log1p` transform or `RobustScaler`.
3. **Handle the class imbalance** — only ~16% of sessions convert. You'll need a documented strategy (e.g. try both `class_weight='balanced'` and SMOTE, and compare results honestly — don't just assume one is better).

**The most important rule of this whole project:**

> ⚠️ Fit your encoders and scalers **only on the training data**, never on the whole dataset before splitting into train/test. Doing it on the full dataset "leaks" information from your test set into training, and makes your model look better than it really is.

**How to make this easy to get right:** Use scikit-learn's `ColumnTransformer` inside a `Pipeline`. This is a built-in scikit-learn tool that handles the "fit only on training data" rule automatically for you, and keeps your numeric and categorical processing separate and organized in one object.

Rough shape of what you're building:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

preprocessor = ColumnTransformer(transformers=[
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
    ("numeric", RobustScaler(), numeric_columns),
])

pipeline = Pipeline(steps=[
    ("preprocessing", preprocessor),
    # you'll add your model here in Week 2
])
```

Once built, **save this pipeline object** with joblib — you'll reuse the exact same fitted pipeline later in your API, so predictions at serving time match training time exactly.

✅ **Done when:**
- Preprocessing is one reusable pipeline object, not scattered notebook cells.
- All four ID columns are treated as categories, not numbers.
- You can explain, in one sentence, how you're handling the class imbalance.
- The fitted pipeline is saved to disk.

---

## Week 1 Checklist — Quick Reference

- [ ] Project folder structure created, Git initialized
- [ ] `config.yaml` created with all key settings
- [ ] Logging set up, no stray `print()` in `src/`
- [ ] Data loader written, validates columns, fails clearly on bad input
- [ ] Thin end-to-end slice working (CSV → rough model → saved → one prediction)
- [ ] EDA notebook complete with one-line takeaways under each chart
- [ ] Decided & written down: is `PageValues` fair to use?
- [ ] Real preprocessing pipeline built with `ColumnTransformer` + `Pipeline`
- [ ] Confirmed: encoders/scalers fit only on training data, never full dataset
- [ ] Fitted pipeline saved to disk

---

## A Few Words of Encouragement

This week feels like a lot of "setup" and not much "AI." That's normal — and it's the part most beginners skip, which is exactly why it's worth doing properly. A messy project with a great model scores worse than a clean, well-organized project with a mediocre one, because the organization is what a real employer is actually evaluating.

If something breaks — and it will — that's expected. Get the rough version working first, then improve it. Don't aim for perfect on the first pass.

Good luck with Week 1! 🎯
