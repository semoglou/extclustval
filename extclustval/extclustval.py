import numpy as np
from functools import cached_property
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    rand_score,
    v_measure_score,
)

class ClusterScore:
    """

    External validation scores for clustering.

    Parameters
    ----------

    - y_true : array-like
        Ground-truth class labels.

    - y_pred : array-like
        Predicted cluster labels.

    Notes
    -----
    ClusterScore stores copies of y_true and y_pred at initialization.
    It also uses cached properties, so if you want to evaluate different
    labels, create a new ClusterScore object.

    """

    def __init__(self, y_true, y_pred):
        self.y_true = np.asarray(y_true).copy()
        self.y_pred = np.asarray(y_pred).copy()

        self._validate_inputs()

    def _validate_inputs(self):
        if self.y_true.ndim != 1:
            raise ValueError("y_true must be one-dimensional.")

        if self.y_pred.ndim != 1:
            raise ValueError("y_pred must be one-dimensional.")

        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

        if len(self.y_true) == 0:
            raise ValueError("y_true and y_pred must not be empty.")

    @cached_property
    def ari(self):
        """Adjusted Rand Index."""
        return float(adjusted_rand_score(self.y_true, self.y_pred))

    @cached_property
    def ri(self):
        """Rand Index."""
        return float(rand_score(self.y_true, self.y_pred))

    @cached_property
    def nmi(self):
        """Normalized Mutual Information."""
        return float(normalized_mutual_info_score(self.y_true, self.y_pred))

    @cached_property
    def ami(self):
        """Adjusted Mutual Information."""
        return float(adjusted_mutual_info_score(self.y_true, self.y_pred))

    @cached_property
    def homogeneity(self):
        """Homogeneity score."""
        return float(homogeneity_score(self.y_true, self.y_pred))

    @cached_property
    def completeness(self):
        """Completeness score."""
        return float(completeness_score(self.y_true, self.y_pred))

    @cached_property
    def v_measure(self):
        """V-measure score."""
        return float(v_measure_score(self.y_true, self.y_pred))

    @cached_property
    def fmi(self):
        """Fowlkes-Mallows Index."""
        return float(fowlkes_mallows_score(self.y_true, self.y_pred))

    @cached_property
    def true_labels(self):
        """Unique ground-truth labels."""
        return np.unique(self.y_true)

    @cached_property
    def pred_labels(self):
        """Unique predicted cluster labels."""
        return np.unique(self.y_pred)

    @cached_property
    def contingency_matrix(self):
        """
        Contingency matrix.

        Rows correspond to ground-truth classes.
        Columns correspond to predicted clusters.
        """
        true_to_idx = {label: idx for idx, label in enumerate(self.true_labels)}
        pred_to_idx = {label: idx for idx, label in enumerate(self.pred_labels)}

        matrix = np.zeros(
            (len(self.true_labels), len(self.pred_labels)),
            dtype=np.int64,
        )

        for true_label, pred_label in zip(self.y_true, self.y_pred):
            i = true_to_idx[true_label]
            j = pred_to_idx[pred_label]
            matrix[i, j] += 1

        return matrix

    @cached_property
    def purity(self):
        """
        Purity score.

        Warning
        -------
        Purity is biased toward solutions with many clusters.
        """
        cm = self.contingency_matrix
        return float(np.sum(np.max(cm, axis=0)) / np.sum(cm))

    @cached_property
    def inverse_purity(self):
        """Inverse purity score."""
        cm = self.contingency_matrix
        return float(np.sum(np.max(cm, axis=1)) / np.sum(cm))

    @cached_property
    def clustering_accuracy(self):
        """
        Clustering accuracy after optimal Hungarian matching.

        This is most appropriate when the number of predicted clusters roughly
        matches the number of ground-truth classes.
        """
        cm = self.contingency_matrix

        row_indices, col_indices = linear_sum_assignment(-cm)

        matched = cm[row_indices, col_indices].sum()
        total = cm.sum()

        return float(matched / total)

    @cached_property
    def acc(self):
        """Alias for clustering_accuracy."""
        return self.clustering_accuracy

    @cached_property
    def pairwise_precision(self):
        """Pairwise precision."""
        return self._pairwise_scores["precision"]

    @cached_property
    def pairwise_recall(self):
        """Pairwise recall."""
        return self._pairwise_scores["recall"]

    @cached_property
    def pairwise_f1(self):
        """Pairwise F1 score."""
        return self._pairwise_scores["f1"]

    @cached_property
    def _pairwise_scores(self):
        """
        Pairwise precision, recall, and F1.

        Computed from the contingency matrix, not from an n x n pair matrix.
        """
        cm = self.contingency_matrix

        tp = np.sum(self._comb2(cm))
        pred_pairs = np.sum(self._comb2(np.sum(cm, axis=0)))
        true_pairs = np.sum(self._comb2(np.sum(cm, axis=1)))

        fp = pred_pairs - tp
        fn = true_pairs - tp

        precision = self._safe_divide(tp, tp + fp)
        recall = self._safe_divide(tp, tp + fn)
        f1 = self._safe_f1(precision, recall)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    @cached_property
    def bcubed_precision(self):
        """BCubed precision."""
        return self._bcubed_scores["precision"]

    @cached_property
    def bcubed_recall(self):
        """BCubed recall."""
        return self._bcubed_scores["recall"]

    @cached_property
    def bcubed_f1(self):
        """BCubed F1 score."""
        return self._bcubed_scores["f1"]

    @cached_property
    def _bcubed_scores(self):
        """
        BCubed precision, recall, and F1.

        Computed from the contingency matrix.
        """
        cm = self.contingency_matrix.astype(float)

        cluster_sizes = np.sum(cm, axis=0)
        class_sizes = np.sum(cm, axis=1)

        precision_terms = np.divide(
            cm**2,
            cluster_sizes,
            out=np.zeros_like(cm),
            where=cluster_sizes > 0,
        )

        recall_terms = np.divide(
            cm**2,
            class_sizes[:, None],
            out=np.zeros_like(cm),
            where=class_sizes[:, None] > 0,
        )

        total = np.sum(cm)

        precision = np.sum(precision_terms) / total
        recall = np.sum(recall_terms) / total
        f1 = self._safe_f1(precision, recall)

        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    @cached_property
    def scores(self):
        """Return all scores as a dictionary."""
        return {
            "ari": self.ari,
            "ri": self.ri,
            "nmi": self.nmi,
            "ami": self.ami,
            "homogeneity": self.homogeneity,
            "completeness": self.completeness,
            "v_measure": self.v_measure,
            "fmi": self.fmi,
            "purity": self.purity,
            "inverse_purity": self.inverse_purity,
            "clustering_accuracy": self.clustering_accuracy,
            "acc": self.acc,
            "pairwise_precision": self.pairwise_precision,
            "pairwise_recall": self.pairwise_recall,
            "pairwise_f1": self.pairwise_f1,
            "bcubed_precision": self.bcubed_precision,
            "bcubed_recall": self.bcubed_recall,
            "bcubed_f1": self.bcubed_f1,
        }

    def to_dict(self):
        """Return all scores as a dictionary."""
        return dict(self.scores)

    @staticmethod
    def _comb2(x):
        """Number of unordered pairs from x items."""
        return x * (x - 1) / 2

    @staticmethod
    def _safe_divide(numerator, denominator):
        if denominator == 0:
            return 0.0
        return numerator / denominator

    @staticmethod
    def _safe_f1(precision, recall):
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
