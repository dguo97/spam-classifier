import joblib
import numpy as np
import pandas as pd
import pytest

import fraud_detection as fd

FEATURES = {
    "Transaction_Amount",
    "Merchant_Category",
    "Transaction_Type",
    "Is_Weekend",
    "Hour",
    "Txns_Last_Hour",
    "Distance_From_Home_Km",
}


@pytest.fixture(scope="module")
def data():
    return fd.generate_synthetic_data(5000, seed=0)


@pytest.fixture(scope="module")
def model(data):
    X, y = data
    return fd.HybridFraudModel(n_estimators=50, seed=0).fit(X, y)


# ---- data ---------------------------------------------------------------


def test_synthetic_data_has_expected_shape_and_fraud_rate(data):
    X, y = data
    assert len(X) == len(y) == 5000
    assert set(X.columns) == FEATURES
    assert 0.015 < y.mean() < 0.05
    assert X["Hour"].between(0, 23).all()
    assert (X["Txns_Last_Hour"] >= 1).all()
    assert (X["Distance_From_Home_Km"] >= 0).all()


def test_synthetic_data_is_reproducible_per_seed():
    a, _ = fd.generate_synthetic_data(500, seed=7)
    b, _ = fd.generate_synthetic_data(500, seed=7)
    c, _ = fd.generate_synthetic_data(500, seed=8)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_fraud_looks_different_from_normal_but_overlaps(data):
    X, y = data
    fraud, normal = X[y == 1], X[y == 0]
    assert fraud["Distance_From_Home_Km"].median() > 3 * normal["Distance_From_Home_Km"].median()
    assert fraud["Txns_Last_Hour"].mean() > normal["Txns_Last_Hour"].mean()
    assert (fraud["Hour"] < 6).mean() > 3 * (normal["Hour"] < 6).mean()
    # classes overlap: plenty of fraud looks ordinary
    assert (fraud["Distance_From_Home_Km"] < normal["Distance_From_Home_Km"].quantile(0.9)).mean() > 0.3


# ---- model --------------------------------------------------------------


def test_fit_rejects_single_class_labels(data):
    X, _ = data
    with pytest.raises(ValueError, match="only one class"):
        fd.HybridFraudModel().fit(X, np.zeros(len(X), dtype=int))


def test_probabilities_are_valid(model, data):
    X, _ = data
    proba = model.predict_proba(X.head(50))
    assert proba.shape == (50,)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_probabilities_are_calibrated_on_fresh_data(model):
    X_new, y_new = fd.generate_synthetic_data(30000, seed=123)
    proba = model.predict_proba(X_new)
    assert abs(proba.mean() - y_new.mean()) < 0.01
    # Only compare bins with enough transactions to give a stable observed rate.
    mean_pred, observed = fd.calibration_points(y_new, proba, min_count=60)
    assert len(mean_pred) >= 4
    assert np.max(np.abs(mean_pred - observed)) < 0.2


def test_calibration_makes_probabilities_closer_to_observed_rate(data):
    X, y = data
    raw = fd.HybridFraudModel(n_estimators=50, seed=0, calibrate=False).fit(X, y)
    X_new, y_new = fd.generate_synthetic_data(8000, seed=321)
    cal_gap = abs(model_mean(fd.HybridFraudModel(n_estimators=50, seed=0).fit(X, y), X_new) - y_new.mean())
    raw_gap = abs(model_mean(raw, X_new) - y_new.mean())
    assert cal_gap <= raw_gap


def model_mean(m, X):
    return float(m.predict_proba(X).mean())


def test_probabilities_never_claim_certainty(model, data):
    X, _ = data
    proba = model.predict_proba(X)
    assert proba.min() >= fd.PROBABILITY_FLOOR
    assert proba.max() <= 1 - fd.PROBABILITY_FLOOR


def test_baseline_has_no_anomaly_score(data):
    X, y = data
    baseline = fd.HybridFraudModel(use_anomaly_score=False, n_estimators=30, seed=0).fit(X, y)
    assert baseline.predict_proba(X.head(5)).shape == (5,)
    with pytest.raises(RuntimeError):
        baseline.anomaly_scores(X.head(5))


def test_risky_transaction_scores_higher_than_safe_one(model):
    risky = fd.score_transaction(
        model, 600, "Electronics", "Online", True, hour=3, txns_last_hour=4, distance_km=400
    )
    safe = fd.score_transaction(model, 12, "Grocery", "In-Store", False, hour=13, distance_km=3)
    assert risky["probability"] > safe["probability"]
    assert risky["probability"] > 0.5 > safe["probability"]
    assert risky["is_fraud"] and not safe["is_fraud"]


def test_new_features_change_the_score(model):
    base = dict(amount=200, category="Retail", trans_type="Online", is_weekend=False)
    near = fd.score_transaction(model, **base, distance_km=2, hour=14, txns_last_hour=1)
    far = fd.score_transaction(model, **base, distance_km=600, hour=3, txns_last_hour=5)
    assert far["probability"] > near["probability"]


def test_threshold_controls_verdict(model):
    args = (model, 12, "Grocery", "In-Store", 0)
    assert fd.score_transaction(*args, threshold=0.0)["is_fraud"]
    assert not fd.score_transaction(*args, threshold=1.0)["is_fraud"]


def test_default_threshold_comes_from_the_model(data):
    X, y = data
    m = fd.HybridFraudModel(n_estimators=30, seed=0).fit(X, y)
    m.threshold_ = 0.07
    assert fd.score_transaction(m, 12, "Grocery", "In-Store")["threshold"] == 0.07
    assert fd.score_transaction(m, 12, "Grocery", "In-Store", threshold=0.4)["threshold"] == 0.4


def test_input_is_validated_and_case_insensitive(model):
    result = fd.score_transaction(model, 50, "travel", " ONLINE ", 0)
    assert (result["category"], result["type"]) == ("Travel", "Online")
    bad_inputs = [
        dict(amount=50, category="Furniture", trans_type="Online"),
        dict(amount=-5, category="Travel", trans_type="Online"),
        dict(amount=float("nan"), category="Travel", trans_type="Online"),
        dict(amount=50, category="Travel", trans_type="Online", hour=24),
        dict(amount=50, category="Travel", trans_type="Online", hour=-1),
        dict(amount=50, category="Travel", trans_type="Online", hour=3.5),
        dict(amount=50, category="Travel", trans_type="Online", txns_last_hour=0),
        dict(amount=50, category="Travel", trans_type="Online", distance_km=-1),
        dict(amount=50, category="Travel", trans_type="Online", distance_km=float("inf")),
    ]
    for kwargs in bad_inputs:
        with pytest.raises(ValueError):
            fd.score_transaction(model, **kwargs)


# ---- tuning and thresholds ----------------------------------------------


def test_expected_cost_counts_both_error_types():
    y_true = [1, 1, 0, 0, 0]
    y_pred = [1, 0, 1, 1, 0]  # one miss, two false alarms
    assert fd.expected_cost(y_true, y_pred, fn_cost=10, fp_cost=1) == 12
    assert fd.expected_cost(y_true, y_pred, fn_cost=1, fp_cost=5) == 11


def test_select_min_samples_leaf_returns_a_candidate(data):
    X, y = data
    leaf = fd.select_min_samples_leaf(X, y, candidates=(1, 5), n_splits=2, n_estimators=10, seed=0)
    assert leaf in (1, 5)


def test_select_threshold_is_valid_and_responds_to_costs(data):
    X, y = data
    kwargs = dict(seed=0, n_estimators=40)
    cheap_miss = fd.select_threshold(X, y, fn_cost=1, fp_cost=1, **kwargs)
    costly_miss = fd.select_threshold(X, y, fn_cost=50, fp_cost=1, **kwargs)
    assert 0 < costly_miss < 1 and 0 < cheap_miss < 1
    assert costly_miss <= cheap_miss  # missing fraud is worse -> flag more readily


# ---- saving and loading -------------------------------------------------


def test_save_and_load_roundtrip(model, data, tmp_path):
    X, _ = data
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    loaded = joblib.load(path)
    np.testing.assert_allclose(loaded.predict_proba(X.head(20)), model.predict_proba(X.head(20)))


def test_load_or_train_saves_sets_threshold_then_reuses(tmp_path):
    path = tmp_path / "model.joblib"
    first = fd.load_or_train(path, n_samples=3000)
    assert path.exists() and 0 < first.threshold_ < 1
    mtime = path.stat().st_mtime_ns
    second = fd.load_or_train(path, n_samples=3000)
    assert path.stat().st_mtime_ns == mtime
    assert second.threshold_ == first.threshold_


def test_load_or_train_recovers_from_corrupt_or_outdated_file(tmp_path):
    path = tmp_path / "model.joblib"
    path.write_bytes(b"not a model")
    assert isinstance(fd.load_or_train(path, n_samples=2000), fd.HybridFraudModel)

    old = fd.HybridFraudModel(n_estimators=10, seed=0)
    old.version_ = 1
    joblib.dump(old, path)
    fresh = fd.load_or_train(path, n_samples=2000)
    assert fresh.version_ == fd.MODEL_VERSION


# ---- evaluation ---------------------------------------------------------


def test_evaluate_reports_sensible_metrics():
    X, y = fd.generate_synthetic_data(6000, seed=3)
    evaluation = fd.evaluate(X, y, n_estimators=50, seed=3, tune=False)
    results = evaluation["results"]
    assert list(results) == [fd.ISOLATION_ONLY, fd.FOREST_ONLY, fd.HYBRID]
    for r in results.values():
        for key in ("precision", "recall", "f1", "pr_auc", "roc_auc"):
            assert 0.0 <= r[key] <= 1.0
        assert r["cost_per_1k"] >= 0
    assert results[fd.ISOLATION_ONLY]["threshold"] is None
    assert 0 < results[fd.HYBRID]["threshold"] < 1
    # A trained model must clearly beat the "no skill" PR-AUC (= fraud rate).
    assert results[fd.HYBRID]["pr_auc"] > evaluation["y_test"].mean() * 5


def test_evaluate_uses_a_fixed_threshold_when_given():
    X, y = fd.generate_synthetic_data(3000, seed=5)
    evaluation = fd.evaluate(X, y, n_estimators=30, seed=5, tune=False, threshold=0.4)
    assert evaluation["results"][fd.HYBRID]["threshold"] == 0.4
    assert evaluation["results"][fd.FOREST_ONLY]["threshold"] == 0.4


def test_evaluate_tuning_selects_a_candidate_leaf():
    X, y = fd.generate_synthetic_data(3000, seed=6)
    evaluation = fd.evaluate(X, y, n_estimators=20, seed=6, tune=True)
    assert evaluation["min_samples_leaf"] in fd.DEFAULT_LEAF_CANDIDATES


def test_evaluate_seeds_runs_each_seed_on_fresh_data(capsys):
    evaluations = fd.evaluate_seeds(2, n_samples=3000, first_seed=10, n_estimators=20, tune=False)
    assert len(evaluations) == 2
    assert not np.array_equal(evaluations[0]["y_test"], evaluations[1]["y_test"])
    fd.print_seed_report(evaluations)
    out = capsys.readouterr().out
    assert "over 2 seeds" in out and "of 2 seeds" in out and "±" in out


def test_calibration_points_ignore_sparse_bins():
    proba = np.array([0.001] * 50 + [0.9] * 3)
    y = np.array([0] * 50 + [1, 1, 0])
    mean_pred, observed = fd.calibration_points(y, proba, min_count=10)
    assert len(mean_pred) == 1  # the 3-sample high bin is dropped
    assert observed[0] == 0.0


def test_save_plots_writes_png_files(tmp_path):
    pytest.importorskip("matplotlib")
    X, y = fd.generate_synthetic_data(4000, seed=4)
    evaluation = fd.evaluate(X, y, n_estimators=30, seed=4, tune=False)
    paths = fd.save_plots(evaluation, tmp_path)
    assert [p.name for p in paths] == ["confusion_matrix.png", "pr_curve.png", "calibration.png"]
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


# ---- interactive prompt and CLI -----------------------------------------


def test_interactive_prompt_reprompts_and_uses_defaults(model, monkeypatch, capsys):
    answers = iter(
        [
            "abc", "-5", "950",             # amount: two bad, then good
            "furniture", "electronics",     # category
            "carrier pigeon", "online",     # type
            "maybe", "y",                   # weekend
            "25", "3",                      # hour: bad, then good
            "0", "4",                       # transactions: bad, then good
            "",                             # distance: default
        ]
    )
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    result = fd.prompt_user_transaction(model)
    out = capsys.readouterr().out
    assert out.count("⚠️") == 7
    assert "VERDICT" in out
    assert (result["amount"], result["category"], result["type"], result["weekend"]) == (
        950.0,
        "Electronics",
        "Online",
        True,
    )
    assert (result["hour"], result["txns_last_hour"], result["distance_km"]) == (3, 4, fd.DEFAULT_DISTANCE_KM)


def test_interactive_prompt_optional_questions_can_all_be_skipped(model, monkeypatch):
    answers = iter(["20", "grocery", "in-store", "n", "", "", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    result = fd.prompt_user_transaction(model)
    assert (result["hour"], result["txns_last_hour"], result["distance_km"]) == (
        fd.DEFAULT_HOUR,
        fd.DEFAULT_TXNS_LAST_HOUR,
        fd.DEFAULT_DISTANCE_KM,
    )


def test_cli_one_shot_check(tmp_path, capsys):
    code = fd.main(
        [
            "--amount", "900",
            "--category", "electronics",
            "--type", "online",
            "--weekend",
            "--hour", "2",
            "--txns-last-hour", "4",
            "--distance-km", "500",
            "--model-path", str(tmp_path / "m.joblib"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert "Calculated Fraud Probability" in out
    assert "Distance from home: 500.0 km" in out


def test_cli_rejects_partial_or_invalid_flags(tmp_path):
    model_path = str(tmp_path / "m.joblib")
    with pytest.raises(SystemExit):
        fd.main(["check", "--amount", "10", "--model-path", model_path])
    with pytest.raises(SystemExit):
        fd.main(["check", "--txns-last-hour", "0"])
    with pytest.raises(SystemExit):
        fd.main(["check", "--threshold", "1.5"])


def test_cli_evaluate_runs_on_synthetic_data(tmp_path, capsys):
    pytest.importorskip("matplotlib")
    code = fd.main(
        ["evaluate", "--n-samples", "3000", "--no-tune", "--plots", "--out-dir", str(tmp_path)]
    )
    out = capsys.readouterr().out
    assert code == 0
    assert fd.HYBRID in out and "Classification report" in out and "Cost/1k" in out
    for name in ("confusion_matrix.png", "pr_curve.png", "calibration.png"):
        assert (tmp_path / name).exists()


def test_cli_evaluate_with_several_seeds(capsys):
    code = fd.main(["evaluate", "--n-samples", "2500", "--no-tune", "--seeds", "2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "over 2 seeds" in out
