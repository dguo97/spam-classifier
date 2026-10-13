"""Credit card fraud detection with a hybrid anomaly-score + classifier model.

An Isolation Forest scores how unusual each transaction looks (unsupervised),
and that score is fed as an extra feature into a Random Forest classifier whose
probabilities are then calibrated.

Two ways to use it:

* ``check``    - score a single transaction (interactive prompts or flags),
                 using a model trained on synthetic data.
* ``evaluate`` - train/test evaluation with a classification report, confusion
                 matrices, precision-recall and calibration curves, optionally
                 repeated over several random seeds.

Everything runs on synthetic data generated on the fly, so there is nothing to
download and any transaction is easy to type in and test.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler

VALID_CATEGORIES = ["Grocery", "Electronics", "Travel", "Dining", "Retail"]
VALID_TYPES = ["In-Store", "Online", "ATM"]
SCALED_COLUMNS = ("Transaction_Amount", "Distance_From_Home_Km")
DEFAULT_MODEL_PATH = "fraud_model.joblib"
MODEL_VERSION = 2  # bump when the saved model's inputs or structure change

# Used when the user does not give these optional details.
DEFAULT_HOUR = 12
DEFAULT_TXNS_LAST_HOUR = 1
DEFAULT_DISTANCE_KM = 5.0

# A missed fraud is assumed to cost this many times more than a false alarm.
DEFAULT_FN_COST = 10.0
DEFAULT_FP_COST = 1.0
PROBABILITY_FLOOR = 0.001  # reported probabilities stay within [0.1%, 99.9%]
DEFAULT_LEAF_CANDIDATES = (1, 3, 5, 10)
DEFAULT_MIN_SAMPLES_LEAF = 3

ISOLATION_ONLY = "Isolation Forest only"
FOREST_ONLY = "Random Forest only"
HYBRID = "Hybrid (IF score + RF)"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------


def generate_synthetic_data(
    n_samples: int = 20000, fraud_rate: float = 0.03, seed: int = 42
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate synthetic transactions. Returns ``(X, y)``.

    Fraud cases are drawn from their own distribution instead of being derived
    from a rule: larger amounts (or small "card-testing" charges), more
    electronics and travel, mostly online, often at night, sometimes in a burst
    of several transactions within an hour, and often far from home. Many fraud
    cases still look completely normal, so the classes overlap and the task is
    not trivially solvable.
    """
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n_samples) < fraud_rate
    n_fraud = int(is_fraud.sum())
    n_legit = n_samples - n_fraud
    legit = ~is_fraud

    amount = np.empty(n_samples)
    amount[legit] = rng.exponential(scale=60, size=n_legit)
    large = rng.random(n_fraud) < 0.7
    amount[is_fraud] = np.where(
        large,
        rng.lognormal(mean=np.log(350), sigma=0.7, size=n_fraud),
        rng.uniform(1, 15, size=n_fraud),
    )

    category = np.empty(n_samples, dtype=object)
    category[legit] = rng.choice(VALID_CATEGORIES, size=n_legit, p=[0.40, 0.15, 0.05, 0.20, 0.20])
    category[is_fraud] = rng.choice(VALID_CATEGORIES, size=n_fraud, p=[0.05, 0.40, 0.25, 0.10, 0.20])

    trans_type = np.empty(n_samples, dtype=object)
    trans_type[legit] = rng.choice(VALID_TYPES, size=n_legit, p=[0.50, 0.40, 0.10])
    trans_type[is_fraud] = rng.choice(VALID_TYPES, size=n_fraud, p=[0.05, 0.80, 0.15])

    weekend = rng.random(n_samples) < np.where(is_fraud, 0.45, 0.30)

    hour = np.empty(n_samples, dtype=int)
    hour[legit] = np.round(rng.normal(14, 4.5, size=n_legit)).astype(int) % 24
    night = rng.random(n_fraud) < 0.45
    hour[is_fraud] = np.where(
        night,
        rng.integers(0, 6, size=n_fraud),
        np.round(rng.normal(14, 4.5, size=n_fraud)).astype(int) % 24,
    )

    txns_last_hour = np.empty(n_samples, dtype=int)
    txns_last_hour[legit] = 1 + rng.poisson(0.4, size=n_legit)
    burst = rng.random(n_fraud) < 0.35
    txns_last_hour[is_fraud] = np.where(
        burst, 2 + rng.poisson(2.5, size=n_fraud), 1 + rng.poisson(0.4, size=n_fraud)
    )

    distance = np.empty(n_samples)
    distance[legit] = rng.lognormal(mean=np.log(6), sigma=1.0, size=n_legit)
    far = rng.random(n_fraud) < 0.5
    distance[is_fraud] = np.where(
        far,
        rng.lognormal(mean=np.log(250), sigma=0.9, size=n_fraud),
        rng.lognormal(mean=np.log(6), sigma=1.0, size=n_fraud),
    )

    X = pd.DataFrame(
        {
            "Transaction_Amount": amount,
            "Merchant_Category": category,
            "Transaction_Type": trans_type,
            "Is_Weekend": weekend.astype(int),
            "Hour": hour,
            "Txns_Last_Hour": txns_last_hour,
            "Distance_From_Home_Km": distance,
        }
    )
    return X, pd.Series(is_fraud.astype(int), name="Fraud_Label")


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """RobustScaler for skewed numeric columns, one-hot for text, rest passed through."""
    scaled_cols = [c for c in X.columns if c in SCALED_COLUMNS]
    cat_cols = [
        c
        for c in X.columns
        if c not in scaled_cols
        and not pd.api.types.is_numeric_dtype(X[c])
        and not pd.api.types.is_bool_dtype(X[c])
    ]
    other_cols = [c for c in X.columns if c not in scaled_cols + cat_cols]

    transformers = []
    if scaled_cols:
        transformers.append(("scaled", RobustScaler(), scaled_cols))
    if cat_cols:
        transformers.append(
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
        )
    if other_cols:
        transformers.append(("rest", "passthrough", other_cols))
    return ColumnTransformer(transformers, sparse_threshold=0)


class HybridFraudModel:
    """Isolation Forest anomaly score + Random Forest classifier.

    * ``use_anomaly_score=False`` gives a plain Random Forest baseline.
    * The anomaly score used to train the classifier is computed out-of-fold:
      each row is scored by an Isolation Forest that never saw it. Otherwise the
      classifier would train on in-sample scores that look different from the
      scores it receives on new data.
    * ``calibrate=True`` wraps the Random Forest in isotonic calibration, so a
      predicted 20% really means roughly 20% of such transactions are fraud
      (class weighting alone inflates the raw probabilities).
    * ``threshold_`` is the decision cut-off used by :meth:`predict`; it starts
      at 0.5 and can be set with :func:`select_threshold`.
    """

    version_ = MODEL_VERSION

    def __init__(
        self,
        use_anomaly_score: bool = True,
        n_estimators: int = 100,
        n_splits: int = 5,
        seed: int = 42,
        min_samples_leaf: int = DEFAULT_MIN_SAMPLES_LEAF,
        calibrate: bool = True,
    ):
        self.use_anomaly_score = use_anomaly_score
        self.n_estimators = n_estimators
        self.n_splits = n_splits
        self.seed = seed
        self.min_samples_leaf = min_samples_leaf
        self.calibrate = calibrate
        self.threshold_ = 0.5

    def _isolation_forest(self, contamination: float) -> IsolationForest:
        return IsolationForest(contamination=contamination, random_state=self.seed, n_jobs=-1)

    def fit(self, X: pd.DataFrame, y) -> "HybridFraudModel":
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("Training labels contain only one class; need both fraud and normal.")

        self.preprocessor_ = make_preprocessor(X)
        Z = self.preprocessor_.fit_transform(X)

        if self.use_anomaly_score:
            # Expected share of anomalies = share of fraud in the training data.
            contamination = float(np.clip(y.mean(), 1e-4, 0.5))
            self.iso_ = self._isolation_forest(contamination).fit(Z)

            oof_scores = np.empty(len(Z))
            folds = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
            for train_idx, val_idx in folds.split(Z):
                fold_iso = self._isolation_forest(contamination).fit(Z[train_idx])
                oof_scores[val_idx] = fold_iso.decision_function(Z[val_idx])
            Z = np.column_stack([Z, oof_scores])

        forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.seed,
            class_weight="balanced_subsample",
            n_jobs=-1,
        )
        if self.calibrate:
            n_pos = int(y.sum())
            cv = max(2, min(5, n_pos, len(y) - n_pos))
            self.clf_ = CalibratedClassifierCV(forest, method="isotonic", cv=cv)
        else:
            self.clf_ = forest
        self.clf_.fit(Z, y)
        return self

    def anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Isolation Forest decision scores (lower = more anomalous)."""
        if not self.use_anomaly_score:
            raise RuntimeError("This model was trained without an anomaly score.")
        return self.iso_.decision_function(self.preprocessor_.transform(X))

    def anomaly_flags(self, X: pd.DataFrame) -> np.ndarray:
        """1 where the Isolation Forest itself flags an anomaly, else 0."""
        if not self.use_anomaly_score:
            raise RuntimeError("This model was trained without an anomaly score.")
        return (self.iso_.predict(self.preprocessor_.transform(X)) == -1).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Fraud probability for each row, kept within [0.1%, 99.9%].

        Isotonic calibration can output exactly 0 or 1 where the training data
        had no counter-examples; clipping avoids claiming certainty.
        """
        Z = self.preprocessor_.transform(X)
        if self.use_anomaly_score:
            Z = np.column_stack([Z, self.iso_.decision_function(Z)])
        fraud_col = int(np.where(self.clf_.classes_ == 1)[0][0])
        return np.clip(self.clf_.predict_proba(Z)[:, fraud_col], PROBABILITY_FLOOR, 1 - PROBABILITY_FLOOR)

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> np.ndarray:
        threshold = self.threshold_ if threshold is None else threshold
        return (self.predict_proba(X) >= threshold).astype(int)


def expected_cost(y_true, y_pred, fn_cost: float = DEFAULT_FN_COST, fp_cost: float = DEFAULT_FP_COST) -> float:
    """Total cost of the mistakes: missed frauds cost ``fn_cost``, false alarms ``fp_cost``."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    false_neg = int(((y_true == 1) & (y_pred == 0)).sum())
    false_pos = int(((y_true == 0) & (y_pred == 1)).sum())
    return fn_cost * false_neg + fp_cost * false_pos


def select_min_samples_leaf(
    X: pd.DataFrame,
    y,
    candidates=DEFAULT_LEAF_CANDIDATES,
    n_splits: int = 3,
    n_estimators: int = 50,
    seed: int = 42,
) -> int:
    """Pick ``min_samples_leaf`` by cross-validated PR-AUC on the given data only.

    Pass training data only, never the test set.
    """
    y = np.asarray(y).astype(int)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    best_leaf, best_score = candidates[0], -1.0
    for leaf in candidates:
        scores = []
        for train_idx, val_idx in folds.split(X, y):
            model = HybridFraudModel(
                n_estimators=n_estimators,
                n_splits=3,
                seed=seed,
                min_samples_leaf=leaf,
                calibrate=False,
            ).fit(X.iloc[train_idx], y[train_idx])
            scores.append(average_precision_score(y[val_idx], model.predict_proba(X.iloc[val_idx])))
        mean_score = float(np.mean(scores))
        if mean_score > best_score + 1e-9:
            best_leaf, best_score = leaf, mean_score
    return int(best_leaf)


def select_threshold(
    X: pd.DataFrame,
    y,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    seed: int = 42,
    **model_kwargs,
) -> float:
    """Choose the cut-off with the lowest expected cost on a held-out part of ``X``.

    Fits a model on 75% of the data and scans thresholds on the other 25%. Pass
    training data only, never the test set. With well-calibrated probabilities
    the best cut-off is close to ``fp_cost / (fp_cost + fn_cost)``.
    """
    y = np.asarray(y).astype(int)
    X_fit, X_val, y_fit, y_val = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    model = HybridFraudModel(seed=seed, **model_kwargs).fit(X_fit, y_fit)
    proba = model.predict_proba(X_val)
    grid = np.linspace(0.01, 0.99, 99)
    costs = np.array([expected_cost(y_val, proba >= t, fn_cost, fp_cost) for t in grid])
    best = np.flatnonzero(np.isclose(costs, costs.min()))
    return float(grid[best[len(best) // 2]])  # middle of the best range, for stability


def load_or_train(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    retrain: bool = False,
    n_samples: int = 20000,
    seed: int = 42,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
) -> HybridFraudModel:
    """Load a saved model, or train one on synthetic data and save it.

    The decision threshold is chosen to minimise expected cost (see
    :func:`select_threshold`). Only load model files you created yourself:
    joblib files are pickles.
    """
    path = Path(model_path)
    if path.exists() and not retrain:
        try:
            model = joblib.load(path)
            if isinstance(model, HybridFraudModel) and getattr(model, "version_", 0) == MODEL_VERSION:
                return model
            print(f"{path} was saved by a different version; retraining.")
        except Exception as exc:  # e.g. saved with an incompatible scikit-learn
            print(f"Could not load {path} ({exc}); retraining.")

    print("Training model on synthetic data...")
    X, y = generate_synthetic_data(n_samples, seed=seed)
    threshold = select_threshold(X, y, fn_cost, fp_cost, seed=seed)
    model = HybridFraudModel(seed=seed).fit(X, y)
    model.threshold_ = threshold
    joblib.dump(model, path)
    return model


# --------------------------------------------------------------------------
# Scoring a single transaction
# --------------------------------------------------------------------------


def normalize_choice(value: str, options: list[str], label: str) -> str:
    """Match ``value`` to one of ``options`` ignoring case and surrounding spaces."""
    lookup = {option.lower(): option for option in options}
    key = str(value).strip().lower()
    if key not in lookup:
        raise ValueError(f"Invalid {label} {value!r}. Please choose from: {', '.join(options)}")
    return lookup[key]


def score_transaction(
    model: HybridFraudModel,
    amount: float,
    category: str,
    trans_type: str,
    is_weekend: int | bool = False,
    *,
    hour: int = DEFAULT_HOUR,
    txns_last_hour: int = DEFAULT_TXNS_LAST_HOUR,
    distance_km: float = DEFAULT_DISTANCE_KM,
    threshold: float | None = None,
) -> dict:
    """Score one transaction. Raises ValueError on invalid input.

    ``threshold`` defaults to the model's own cost-optimised cut-off.
    """
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be a non-negative number.")
    if int(hour) != hour or not 0 <= hour <= 23:
        raise ValueError("Hour must be a whole number from 0 to 23.")
    if int(txns_last_hour) != txns_last_hour or txns_last_hour < 1:
        raise ValueError("Transactions in the last hour must be a whole number, at least 1.")
    if not math.isfinite(distance_km) or distance_km < 0:
        raise ValueError("Distance from home must be a non-negative number.")
    category = normalize_choice(category, VALID_CATEGORIES, "category")
    trans_type = normalize_choice(trans_type, VALID_TYPES, "transaction type")
    if threshold is None:
        threshold = getattr(model, "threshold_", 0.5)

    row = pd.DataFrame(
        {
            "Transaction_Amount": [float(amount)],
            "Merchant_Category": [category],
            "Transaction_Type": [trans_type],
            "Is_Weekend": [int(bool(is_weekend))],
            "Hour": [int(hour)],
            "Txns_Last_Hour": [int(txns_last_hour)],
            "Distance_From_Home_Km": [float(distance_km)],
        }
    )
    probability = float(model.predict_proba(row)[0])
    return {
        "amount": float(amount),
        "category": category,
        "type": trans_type,
        "weekend": bool(is_weekend),
        "hour": int(hour),
        "txns_last_hour": int(txns_last_hour),
        "distance_km": float(distance_km),
        "probability": probability,
        "is_fraud": probability >= threshold,
        "threshold": float(threshold),
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 50)
    print("📊 ANALYSIS RESULTS")
    print("=" * 50)
    print(f"• Amount: ${result['amount']:.2f}")
    print(f"• Category: {result['category']}")
    print(f"• Type: {result['type']}")
    print(f"• Weekend: {'Yes' if result['weekend'] else 'No'}")
    print(f"• Hour of day: {result['hour']:02d}:00")
    print(f"• Transactions in the last hour: {result['txns_last_hour']}")
    print(f"• Distance from home: {result['distance_km']:.1f} km")
    print(f"• Calculated Fraud Probability: {result['probability']:.2%}")
    print(f"• Decision threshold: {result['threshold']:.0%}")
    print("-" * 50)
    if result["is_fraud"]:
        print("🚨 VERDICT: High Risk / Potential Fraud Detected!")
    else:
        print("✅ VERDICT: Normal Transaction")
    print("=" * 50 + "\n")


def _parse_amount(raw: str) -> float:
    try:
        amount = float(raw)
    except ValueError:
        raise ValueError("Invalid input. Please enter a valid number.") from None
    if not math.isfinite(amount) or amount < 0:
        raise ValueError("Amount must be a non-negative number. Try again.")
    return amount


def _parse_weekend(raw: str) -> int:
    answer = raw.strip().lower()
    if answer in ("y", "yes"):
        return 1
    if answer in ("n", "no"):
        return 0
    raise ValueError("Please type 'y' for yes or 'n' for no.")


def _parse_hour(raw: str) -> int:
    if not raw:
        return DEFAULT_HOUR
    try:
        hour = int(raw)
    except ValueError:
        raise ValueError("Please enter a whole number from 0 to 23.") from None
    if not 0 <= hour <= 23:
        raise ValueError("Please enter a whole number from 0 to 23.")
    return hour


def _parse_txns(raw: str) -> int:
    if not raw:
        return DEFAULT_TXNS_LAST_HOUR
    try:
        count = int(raw)
    except ValueError:
        raise ValueError("Please enter a whole number, at least 1.") from None
    if count < 1:
        raise ValueError("Please enter a whole number, at least 1.")
    return count


def _parse_distance(raw: str) -> float:
    if not raw:
        return DEFAULT_DISTANCE_KM
    try:
        distance = float(raw)
    except ValueError:
        raise ValueError("Please enter a valid number of kilometres.") from None
    if not math.isfinite(distance) or distance < 0:
        raise ValueError("Distance cannot be negative.")
    return distance


def _ask(prompt: str, parse):
    """Keep asking until ``parse`` accepts the answer."""
    while True:
        try:
            return parse(input(prompt).strip())
        except ValueError as exc:
            print(f"⚠️ {exc}")


def prompt_user_transaction(model: HybridFraudModel, threshold: float | None = None) -> dict:
    """Guide the user step by step, validating every answer.

    The last three questions are optional: press Enter to use the default.
    """
    print("=" * 50)
    print("💳 CREDIT CARD FRAUD DETECTION - INTERACTIVE CHECKER")
    print("=" * 50)
    print("Please enter the transaction details below when prompted.\n")

    amount = _ask("1. Enter Transaction Amount (e.g., 45.50 or 1200.00): $", _parse_amount)

    print(f"\n2. Available Merchant Categories: {', '.join(VALID_CATEGORIES)}")
    category = _ask(
        "Select or type a category from the list above: ",
        lambda raw: normalize_choice(raw, VALID_CATEGORIES, "category"),
    )

    print(f"\n3. Available Transaction Types: {', '.join(VALID_TYPES)}")
    trans_type = _ask(
        "Select transaction type from the list above: ",
        lambda raw: normalize_choice(raw, VALID_TYPES, "transaction type"),
    )

    is_weekend = _ask("\n4. Is this transaction taking place on a weekend? (y/n): ", _parse_weekend)

    print("\nOptional details (press Enter to use the default):")
    hour = _ask(f"5. Hour of day, 0-23 [{DEFAULT_HOUR}]: ", _parse_hour)
    txns = _ask(
        f"6. Transactions on this card in the last hour, including this one [{DEFAULT_TXNS_LAST_HOUR}]: ",
        _parse_txns,
    )
    distance = _ask(
        f"7. Distance from the cardholder's home in km [{DEFAULT_DISTANCE_KM:g}]: ", _parse_distance
    )

    result = score_transaction(
        model,
        amount,
        category,
        trans_type,
        is_weekend,
        hour=hour,
        txns_last_hour=txns,
        distance_km=distance,
        threshold=threshold,
    )
    print_result(result)
    return result


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _summarise(y_true, score, y_pred, fn_cost: float, fp_cost: float, threshold: float | None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, score)
    return {
        "score": score,
        "y_pred": y_pred,
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pr_auc": float(auc(rec_curve, prec_curve)),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "cost_per_1k": expected_cost(y_true, y_pred, fn_cost, fp_cost) / len(y_true) * 1000,
    }


def evaluate(
    X: pd.DataFrame,
    y,
    seed: int = 42,
    n_estimators: int = 100,
    test_size: float = 0.2,
    fn_cost: float = DEFAULT_FN_COST,
    fp_cost: float = DEFAULT_FP_COST,
    threshold: float | None = None,
    tune: bool = True,
) -> dict:
    """Stratified train/test evaluation of three detectors.

    * Isolation Forest only (unsupervised, flags its own top anomalies)
    * Random Forest only (supervised baseline)
    * Hybrid: Random Forest that also receives the Isolation Forest score

    Everything is chosen on the training split only: ``min_samples_leaf`` by
    cross-validation (``tune=True``) and each model's decision threshold by
    minimising expected cost on a held-out part of the training data (unless a
    fixed ``threshold`` is given). The test split is used once, for reporting.
    """
    y = np.asarray(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    leaf = (
        select_min_samples_leaf(X_train, y_train, seed=seed) if tune else DEFAULT_MIN_SAMPLES_LEAF
    )
    common = dict(n_estimators=n_estimators, min_samples_leaf=leaf)

    hybrid = HybridFraudModel(True, seed=seed, **common).fit(X_train, y_train)
    baseline = HybridFraudModel(False, seed=seed, **common).fit(X_train, y_train)

    def cut_off(use_anomaly_score: bool) -> float:
        if threshold is not None:
            return float(threshold)
        return select_threshold(
            X_train, y_train, fn_cost, fp_cost, seed=seed, use_anomaly_score=use_anomaly_score, **common
        )

    results = {
        ISOLATION_ONLY: _summarise(
            y_test,
            -hybrid.anomaly_scores(X_test),
            hybrid.anomaly_flags(X_test),
            fn_cost,
            fp_cost,
            None,
        )
    }
    for name, model, use_score in ((FOREST_ONLY, baseline, False), (HYBRID, hybrid, True)):
        t = cut_off(use_score)
        model.threshold_ = t
        proba = model.predict_proba(X_test)
        results[name] = _summarise(y_test, proba, (proba >= t).astype(int), fn_cost, fp_cost, t)

    return {
        "y_test": y_test,
        "results": results,
        "min_samples_leaf": leaf,
        "fn_cost": fn_cost,
        "fp_cost": fp_cost,
    }


def evaluate_seeds(
    n_seeds: int,
    n_samples: int = 20000,
    first_seed: int = 42,
    **kwargs,
) -> list[dict]:
    """Repeat :func:`evaluate` on freshly generated data for several seeds."""
    evaluations = []
    for seed in range(first_seed, first_seed + n_seeds):
        X, y = generate_synthetic_data(n_samples, seed=seed)
        evaluations.append(evaluate(X, y, seed=seed, **kwargs))
    return evaluations


def print_report(evaluation: dict) -> None:
    y_test = evaluation["y_test"]
    results = evaluation["results"]

    print(
        f"\nTest set: {len(y_test):,} transactions, "
        f"{int(y_test.sum()):,} fraud ({y_test.mean():.3%})"
    )
    print(
        f"Selected min_samples_leaf = {evaluation['min_samples_leaf']} (cross-validated on training data)"
    )
    print(
        f"Cost model: a missed fraud costs {evaluation['fn_cost']:g}, a false alarm costs "
        f"{evaluation['fp_cost']:g}\n"
    )
    print(
        f"{'Model':<24}{'Thresh':>8}{'Precision':>11}{'Recall':>8}{'F1':>7}"
        f"{'PR-AUC':>8}{'ROC-AUC':>9}{'Cost/1k':>9}"
    )
    print("-" * 84)
    for name, r in results.items():
        thresh = f"{r['threshold']:.2f}" if r["threshold"] is not None else "own"
        print(
            f"{name:<24}{thresh:>8}{r['precision']:>11.3f}{r['recall']:>8.3f}{r['f1']:>7.3f}"
            f"{r['pr_auc']:>8.3f}{r['roc_auc']:>9.3f}{r['cost_per_1k']:>9.1f}"
        )
    print("\nThresholds are chosen on training data to minimise expected cost.")
    print("Isolation Forest only uses its own contamination-based cut-off.")
    print("Cost/1k = expected cost per 1,000 transactions (lower is better).")

    print(f"\nClassification report ({HYBRID}):")
    print(
        classification_report(
            y_test,
            results[HYBRID]["y_pred"],
            target_names=["Normal", "Fraud"],
            digits=3,
            zero_division=0,
        )
    )


def print_seed_report(evaluations: list[dict]) -> None:
    """Mean and standard deviation of each metric across seeds, plus head-to-head wins."""
    names = list(evaluations[0]["results"])
    metrics = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1"),
               ("pr_auc", "PR-AUC"), ("roc_auc", "ROC-AUC"), ("cost_per_1k", "Cost/1k")]

    print(f"\nMean ± standard deviation over {len(evaluations)} seeds (fresh data each time)\n")
    print(f"{'Model':<24}" + "".join(f"{label:>16}" for _, label in metrics))
    print("-" * (24 + 16 * len(metrics)))
    for name in names:
        cells = []
        for key, _ in metrics:
            values = np.array([e["results"][name][key] for e in evaluations])
            digits = 1 if key == "cost_per_1k" else 3
            cells.append(f"{values.mean():.{digits}f} ± {values.std():.{digits}f}".rjust(16))
        print(f"{name:<24}" + "".join(cells))

    wins_pr = sum(
        e["results"][HYBRID]["pr_auc"] > e["results"][FOREST_ONLY]["pr_auc"] for e in evaluations
    )
    wins_cost = sum(
        e["results"][HYBRID]["cost_per_1k"] < e["results"][FOREST_ONLY]["cost_per_1k"]
        for e in evaluations
    )
    n = len(evaluations)
    print(f"\nHybrid beats Random Forest only on PR-AUC in {wins_pr} of {n} seeds "
          f"and on cost in {wins_cost} of {n}.")


def calibration_points(
    y_true,
    proba,
    edges=(0.0, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1.0),
    min_count: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean predicted probability and observed fraud rate in fixed probability bins.

    Fraud is rare, so equal-width or equal-count bins leave almost every point
    near zero; these bins are finer at the low end. Bins with fewer than
    ``min_count`` transactions are dropped because their rates are too noisy.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    bin_ids = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, len(edges) - 2)
    mean_pred, observed = [], []
    for b in range(len(edges) - 1):
        in_bin = bin_ids == b
        if in_bin.sum() >= min_count:
            mean_pred.append(proba[in_bin].mean())
            observed.append(y_true[in_bin].mean())
    return np.array(mean_pred), np.array(observed)


def save_plots(evaluation: dict, out_dir: str | Path = ".") -> list[Path]:
    """Save confusion matrices, precision-recall curves and a calibration plot as PNGs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_test = evaluation["y_test"]
    results = evaluation["results"]

    fig, axes = plt.subplots(1, len(results), figsize=(4.5 * len(results), 4))
    for ax, (name, r) in zip(np.atleast_1d(axes), results.items()):
        ConfusionMatrixDisplay.from_predictions(
            y_test,
            r["y_pred"],
            display_labels=["Normal", "Fraud"],
            cmap="Blues",
            colorbar=False,
            ax=ax,
        )
        ax.set_title(name)
    fig.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, r in results.items():
        precision, recall, _ = precision_recall_curve(y_test, r["score"])
        ax.plot(recall, precision, label=f"{name} (PR-AUC {r['pr_auc']:.3f})")
    ax.axhline(y_test.mean(), linestyle="--", color="grey", label="No skill")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    pr_path = out_dir / "pr_curve.png"
    fig.savefig(pr_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfectly calibrated")
    for name in (FOREST_ONLY, HYBRID):
        mean_pred, frac_pos = calibration_points(y_test, results[name]["score"])
        ax.plot(mean_pred, frac_pos, marker="o", label=name)
    ax.set_xlabel("Predicted fraud probability (bin average)")
    ax.set_ylabel("Observed fraud rate")
    ax.set_title("Calibration (test set)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    cal_path = out_dir / "calibration.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)

    return [cm_path, pr_path, cal_path]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def _probability(value: str) -> float:
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError("threshold must be between 0 and 1")
    return number


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def _non_negative_float(value: str) -> float:
    number = float(value)
    if number < 0 or not math.isfinite(number):
        raise argparse.ArgumentTypeError("must be a non-negative number")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fraud_detection.py",
        description="Hybrid Isolation Forest + Random Forest credit card fraud detection.",
        epilog="Running with no command is the same as 'check' (interactive prompts).",
    )
    sub = parser.add_subparsers(dest="command")

    check = sub.add_parser("check", help="score one transaction (prompts if no flags given)")
    check.add_argument("--amount", type=float, help="transaction amount in dollars")
    check.add_argument("--category", help=f"one of: {', '.join(VALID_CATEGORIES)}")
    check.add_argument("--type", dest="trans_type", help=f"one of: {', '.join(VALID_TYPES)}")
    check.add_argument("--weekend", action="store_true", help="transaction happens on a weekend")
    check.add_argument("--hour", type=int, default=DEFAULT_HOUR, help=f"hour of day 0-23 (default {DEFAULT_HOUR})")
    check.add_argument(
        "--txns-last-hour",
        type=_positive_int,
        default=DEFAULT_TXNS_LAST_HOUR,
        help=f"transactions on the card in the last hour, including this one (default {DEFAULT_TXNS_LAST_HOUR})",
    )
    check.add_argument(
        "--distance-km",
        type=_non_negative_float,
        default=DEFAULT_DISTANCE_KM,
        help=f"distance from the cardholder's home in km (default {DEFAULT_DISTANCE_KM:g})",
    )
    check.add_argument(
        "--threshold",
        type=_probability,
        default=None,
        help="fraud cut-off (default: the model's cost-optimised threshold)",
    )
    check.add_argument("--model-path", default=DEFAULT_MODEL_PATH, help="where to save/load the model")
    check.add_argument("--retrain", action="store_true", help="ignore any saved model and retrain")

    ev = sub.add_parser("evaluate", help="train/test evaluation with metrics and plots")
    ev.add_argument("--n-samples", type=_positive_int, default=20000, help="synthetic sample count")
    ev.add_argument("--seeds", type=_positive_int, default=1, help="repeat over this many seeds and report mean ± std")
    ev.add_argument("--seed", type=int, default=42, help="first random seed")
    ev.add_argument(
        "--threshold",
        type=_probability,
        default=None,
        help="use one fixed cut-off for the supervised models instead of choosing by cost",
    )
    ev.add_argument("--fn-cost", type=_non_negative_float, default=DEFAULT_FN_COST, help="cost of a missed fraud")
    ev.add_argument("--fp-cost", type=_non_negative_float, default=DEFAULT_FP_COST, help="cost of a false alarm")
    ev.add_argument("--no-tune", action="store_true", help="skip cross-validated tuning (faster)")
    ev.add_argument("--plots", action="store_true", help="save confusion matrix, PR curve and calibration PNGs")
    ev.add_argument("--out-dir", default=".", help="where to save plots")
    return parser


def _run_check(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    given = [args.amount is not None, args.category is not None, args.trans_type is not None]
    if any(given) and not all(given):
        parser.error("--amount, --category and --type must be given together")

    model = load_or_train(args.model_path, retrain=args.retrain)
    if all(given):
        try:
            result = score_transaction(
                model,
                args.amount,
                args.category,
                args.trans_type,
                args.weekend,
                hour=args.hour,
                txns_last_hour=args.txns_last_hour,
                distance_km=args.distance_km,
                threshold=args.threshold,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print_result(result)
    else:
        prompt_user_transaction(model, args.threshold)
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    print(f"Generating synthetic transactions ({args.n_samples:,} per run) and training models...")
    evaluations = evaluate_seeds(
        args.seeds,
        n_samples=args.n_samples,
        first_seed=args.seed,
        fn_cost=args.fn_cost,
        fp_cost=args.fp_cost,
        threshold=args.threshold,
        tune=not args.no_tune,
    )
    first = evaluations[0]
    print_report(first)
    if args.seeds > 1:
        print_seed_report(evaluations)

    if args.plots:
        for path in save_plots(first, args.out_dir):
            print(f"Saved {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in {"check", "evaluate", "-h", "--help"}:
        argv.insert(0, "check")
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "evaluate":
        return _run_evaluate(args)
    return _run_check(args, parser)


if __name__ == "__main__":
    # Re-import under the real module name so a saved model is pickled as
    # "fraud_detection.HybridFraudModel" rather than "__main__.HybridFraudModel",
    # which could not be loaded from anywhere else.
    import fraud_detection

    raise SystemExit(fraud_detection.main())
