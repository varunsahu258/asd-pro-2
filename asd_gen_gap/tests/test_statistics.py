"""Sanity checks for resampling and paired-comparison statistics."""

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

from asd_gen_gap.stats.bootstrap_ci import bootstrap_ci
from asd_gen_gap.stats.pairwise_tests import delong_test, holm_bonferroni, mcnemar_test


def test_bootstrap_ci_has_hand_checkable_range():
    # Every resampled mean of these four subject-level values is in [0, 1].
    predictions = pd.DataFrame({"subject_id": ["a", "b", "c", "d"], "y_prob": [0.0, 0.0, 1.0, 1.0]})
    point, low, high = bootstrap_ci(
        predictions, lambda frame: frame["y_prob"].mean(), n_resamples=2_000, random_state=3
    )
    assert point == 0.5
    assert 0 <= low < 0.5 < high <= 1
    assert low < 0.1 and high > 0.9


def test_delong_matches_sklearn_auc_and_handles_identical_scores():
    labels = [0, 0, 0, 1, 1, 1]
    first = [0.1, 0.4, 0.35, 0.5, 0.8, 0.9]
    second = [0.2, 0.3, 0.6, 0.45, 0.7, 0.85]
    auc_first, auc_second, p_value = delong_test(labels, first, second)
    assert auc_first == roc_auc_score(labels, first)
    assert auc_second == roc_auc_score(labels, second)
    assert 0 <= p_value <= 1
    _, _, identical_p_value = delong_test(labels, first, first)
    assert identical_p_value == 1.0


def test_mcnemar_matches_continuity_corrected_textbook_formula():
    # Discordant cells are b=10 (A-only correct) and c=20 (B-only correct).
    truth = np.zeros(30, dtype=int)
    first_predictions = np.r_[np.zeros(10, dtype=int), np.ones(20, dtype=int)]
    second_predictions = np.r_[np.ones(10, dtype=int), np.zeros(20, dtype=int)]
    statistic, p_value = mcnemar_test(truth, first_predictions, second_predictions)
    assert statistic == 81 / 30
    assert p_value == chi2.sf(81 / 30, 1)


def test_holm_bonferroni_returns_standard_adjustments():
    adjusted, rejected = holm_bonferroni([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert rejected.tolist() == [True, False, False]
