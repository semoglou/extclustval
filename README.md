# extclustval

A lightweight Python package for external clustering validation metrics.

`extclustval` provides a simple `ClusterScore` class for evaluating clustering results against ground-truth labels.

## Metrics included

### Standard external clustering metrics

| Attribute | Metric |
|---|---|
| `ari` | Adjusted Rand Index |
| `ri` | Rand Index |
| `nmi` | Normalized Mutual Information |
| `ami` | Adjusted Mutual Information |
| `homogeneity` | Homogeneity score |
| `completeness` | Completeness score |
| `v_measure` | V-measure |
| `fmi` | Fowlkes-Mallows Index |

### Additional clustering validation metrics

| Attribute | Metric |
|---|---|
| `purity` | Purity score |
| `inverse_purity` | Inverse purity score |
| `clustering_accuracy` | Hungarian-matched clustering accuracy |

### Pairwise metrics

| Attribute | Metric |
|---|---|
| `pairwise_precision` | Pairwise precision |
| `pairwise_recall` | Pairwise recall |
| `pairwise_f1` | Pairwise F1 score |

### BCubed metrics

| Attribute | Metric |
|---|---|
| `bcubed_precision` | BCubed precision |
| `bcubed_recall` | BCubed recall |
| `bcubed_f1` | BCubed F1 score |

## Installation

You can install `extclustval` from PyPI:

```bash
pip install extclustval
```

## Quick start

```python
from extclustval import ClusterScore

y_true = [0, 0, 1, 1, 2, 2]
y_pred = [1, 1, 0, 0, 2, 2]

score = ClusterScore(y_true, y_pred)

print(score.ari)
print(score.nmi)
print(score.clustering_accuracy)
```

Output:

```text
1.0
1.0
1.0
```

You can also access metrics directly as attributes:

```python
score.ari
score.ri
score.nmi
score.ami
score.homogeneity
score.completeness
score.v_measure
score.fmi
score.purity
score.inverse_purity
score.clustering_accuracy
score.acc
score.pairwise_precision
score.pairwise_recall
score.pairwise_f1
score.bcubed_precision
score.bcubed_recall
score.bcubed_f1
```

## Dictionary output

You can return all scores as a dictionary:

```python
scores = score.to_dict()
print(scores)
```

Example:

```python
{
    "ari": 1.0,
    "ri": 1.0,
    "nmi": 1.0,
    "ami": 1.0,
    "homogeneity": 1.0,
    "completeness": 1.0,
    "v_measure": 1.0,
    "fmi": 1.0,
    "purity": 1.0,
    "inverse_purity": 1.0,
    "clustering_accuracy": 1.0,
    "acc": 1.0,
    "pairwise_precision": 1.0,
    "pairwise_recall": 1.0,
    "pairwise_f1": 1.0,
    "bcubed_precision": 1.0,
    "bcubed_recall": 1.0,
    "bcubed_f1": 1.0,
}
```

## Notes about clustering accuracy

Clustering labels are arbitrary. For example, these two clusterings are equivalent:

```python
[0, 0, 1, 1]
[5, 5, 9, 9]
```

Because of this, `extclustval` computes clustering accuracy using optimal Hungarian matching between predicted clusters and ground-truth classes.

```python
score.clustering_accuracy
```

The short alias `score.acc` is also available and returns the same value.

This metric is most appropriate when the number of predicted clusters roughly matches the number of ground-truth classes.

For general clustering evaluation, adjusted and permutation-invariant metrics such as ARI, AMI, NMI, pairwise F1, and BCubed F1 are often safer to report.

## Notes about purity

Purity is easy to understand, but it is biased toward solutions with many clusters. If each sample is placed in its own cluster, purity can become artificially high.

Use purity together with other metrics such as ARI, AMI, pairwise F1, or BCubed F1.

## Cached properties

`ClusterScore` uses cached properties.

This means each score is computed once and then stored.

```python
score = ClusterScore(y_true, y_pred)

score.ari  # computed once
score.ari  # reused from cache
```

If you want to evaluate different labels, create a new `ClusterScore` object:

```python
score = ClusterScore(y_true, y_pred)

new_score = ClusterScore(y_true, new_y_pred)
```

Do not modify `score.y_true` or `score.y_pred` after creating the object.

## Metric definitions

### Adjusted Rand Index (ARI)

Adjusted Rand Index measures similarity between two partitions by comparing pairs of samples. It is adjusted for chance, so random clusterings tend to score near 0. A perfect match scores 1.

### Rand Index (RI)

Rand Index measures the proportion of sample pairs that are consistently grouped or separated in both the true labels and predicted clusters. It is not adjusted for chance.

### Normalized Mutual Information (NMI)

Normalized Mutual Information measures how much information the predicted clusters and true labels share, normalized to a fixed range. A perfect match scores 1.

### Adjusted Mutual Information (AMI)

Adjusted Mutual Information is a chance-adjusted version of mutual information. It is often safer than NMI when comparing clusterings with different numbers of clusters.

### Homogeneity

Homogeneity measures whether each predicted cluster contains samples from only one ground-truth class.

### Completeness

Completeness measures whether all samples from the same ground-truth class are assigned to the same predicted cluster.

### V-measure

V-measure is the harmonic mean of homogeneity and completeness.

### Fowlkes-Mallows Index (FMI)

Fowlkes-Mallows Index is a pair-counting metric based on the geometric mean of pairwise precision and pairwise recall.

### Purity

Purity measures how much each predicted cluster is dominated by its most common ground-truth class. It is easy to interpret but biased toward many clusters.

### Inverse purity

Inverse purity measures how well each ground-truth class is captured by its best matching predicted cluster.

### Clustering accuracy

Clustering accuracy uses optimal Hungarian matching to align predicted cluster IDs with ground-truth class labels before computing accuracy. This is useful when predicted clusters roughly correspond one-to-one with true classes.

### Pairwise precision

Pairwise precision measures, among all pairs placed in the same predicted cluster, how many also belong to the same ground-truth class.

### Pairwise recall

Pairwise recall measures, among all pairs belonging to the same ground-truth class, how many were placed in the same predicted cluster.

### Pairwise F1

Pairwise F1 is the harmonic mean of pairwise precision and pairwise recall.

### BCubed precision

BCubed precision measures, for each sample, how pure its predicted cluster is with respect to that sample’s true class, then averages over all samples.

### BCubed recall

BCubed recall measures, for each sample, how well its true class is recovered inside its predicted cluster, then averages over all samples.

### BCubed F1

BCubed F1 is the harmonic mean of BCubed precision and BCubed recall.

## Requirements

```text
numpy
scipy
scikit-learn
```

## License

This project is licensed under the [MIT](https://github.com/semoglou/extclustval/blob/main/LICENSE) License.
