# extclustval

<p align="center">
  <a href="https://pypi.org/project/extclustval/"><img src="https://img.shields.io/pypi/v/extclustval.svg?color=blue" alt="PyPI version"></a>&nbsp;&nbsp;
  <a href="https://pypi.org/project/extclustval/"><img src="https://img.shields.io/pypi/pyversions/extclustval.svg" alt="Python versions"></a>&nbsp;&nbsp;
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>&nbsp;&nbsp;
  <a href="https://pepy.tech/project/extclustval"><img src="https://pepy.tech/badge/extclustval" alt="Downloads"></a>
</p>

A lightweight Python package for external clustering validation metrics.

`extclustval` provides a simple `ClusterScore` class for evaluating clustering results against ground-truth labels.

## Metrics included

### Standard external clustering metrics

| Attribute | Metric |
|---|---|
| `ri` | Rand Index |
| `ari` | Adjusted Rand Index |
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

You can install `extclustval` from [PyPI](https://pypi.org/project/extclustval/):

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

### Rand Index (RI)

Rand Index measures how similar two partitions are by looking at all possible pairs of samples.

For every pair of samples, RI checks whether the true labels and predicted clusters agree:

- the pair is in the same true class and also in the same predicted cluster, or
- the pair is in different true classes and also in different predicted clusters.

The score is the fraction of pairs where this agreement happens.

A perfect clustering scores `1.0`. However, RI is not adjusted for chance, so it can sometimes look high even when the clustering is not very meaningful, especially when many sample pairs are easy to separate.

### Adjusted Rand Index (ARI)

Adjusted Rand Index is a chance-adjusted version of the Rand Index.

Like RI, ARI compares pairs of samples and checks whether the true labels and predicted clusters agree. The difference is that ARI corrects for the agreement that would be expected just by random chance.

A perfect clustering scores `1.0`. Random clusterings tend to score near `0.0`. Bad clusterings can score below `0.0`.

ARI is one of the most commonly used external clustering validation metrics because it is permutation-invariant and adjusted for chance.

### Normalized Mutual Information (NMI)

Normalized Mutual Information measures how much information the predicted clusters contain about the true labels.

If knowing a sample’s predicted cluster tells you a lot about its true class, NMI is high. If the predicted clusters and true labels are mostly unrelated, NMI is low.

NMI is normalized so that a perfect match scores `1.0`. It is permutation-invariant, meaning it does not matter which numeric IDs are used for the clusters. However, NMI is not adjusted for chance.

### Adjusted Mutual Information (AMI)

Adjusted Mutual Information is a chance-adjusted version of mutual information.

Like NMI, AMI measures how much information is shared between the predicted clusters and the true labels. Unlike NMI, AMI corrects for the amount of information that would be expected by random cluster assignments.

A perfect clustering scores `1.0`. Random clusterings tend to score near `0.0`.

AMI is useful when comparing clustering results with different numbers of clusters, because it is less likely than NMI to reward structure that appears only by chance.

### Homogeneity

Homogeneity measures whether each predicted cluster contains samples from only one ground-truth class.

A clustering has high homogeneity when its clusters are pure. For example, if one predicted cluster contains only samples from class `A`, that cluster is homogeneous.

Homogeneity penalizes clusters that mix multiple true classes together. However, homogeneity alone does not penalize splitting one true class into many small clusters.

### Completeness

Completeness measures whether all samples from the same ground-truth class are assigned to the same predicted cluster.

A clustering has high completeness when each true class is mostly captured by one cluster. For example, if all samples from class `A` are placed in the same predicted cluster, completeness is high for that class.

Completeness penalizes splitting a true class across multiple clusters. However, completeness alone does not strongly penalize merging different true classes into the same cluster.

### V-measure

V-measure combines homogeneity and completeness into one score.

It is the harmonic mean of homogeneity and completeness, so it rewards clusterings that are both class-pure and class-complete.

A perfect clustering scores `1.0`. V-measure is useful when you want a single score that balances over-splitting and over-merging.

### Fowlkes-Mallows Index (FMI)

Fowlkes-Mallows Index is a pair-based clustering metric.

It looks at pairs of samples and compares which pairs are placed together in the predicted clustering and which pairs truly belong together according to the ground-truth labels.

FMI is the geometric mean of pairwise precision and pairwise recall. A perfect clustering scores `1.0`.

It is useful when you want a score based on pairwise grouping behavior.

### Purity

Purity measures how much each predicted cluster is dominated by a single ground-truth class.

For each predicted cluster, purity finds the most common true class inside that cluster. It then sums these dominant-class counts across all clusters and divides by the total number of samples.

Purity is easy to understand: high purity means clusters mostly contain samples from one class.

However, purity is biased toward many clusters. If every sample is placed in its own cluster, purity becomes perfect, even though the clustering may not be useful.

### Inverse purity

Inverse purity is the class-oriented counterpart of purity.

Instead of asking whether each predicted cluster is dominated by one true class, inverse purity asks whether each true class is well captured by one predicted cluster.

For each ground-truth class, it finds the predicted cluster that contains the largest number of samples from that class. These counts are summed and divided by the total number of samples.

Inverse purity rewards clusterings that avoid splitting true classes across many clusters.

### Clustering accuracy

Clustering accuracy compares predicted cluster labels with ground-truth labels after aligning cluster IDs to class IDs.

Raw classification accuracy is not appropriate for clustering because cluster labels are arbitrary. For example, cluster `0` and cluster `5` could represent the same group.

To handle this, `extclustval` uses optimal Hungarian matching to find the best one-to-one mapping between predicted clusters and true classes. It then computes the fraction of samples that are correctly matched under that mapping.

This metric is most appropriate when the predicted clusters roughly correspond one-to-one with the ground-truth classes. It can be misleading when the number of clusters and classes differs a lot, or when the clustering intentionally splits or merges classes.

### Pairwise precision

Pairwise precision measures how reliable predicted same-cluster decisions are.

It looks at all sample pairs that were placed in the same predicted cluster. Among those pairs, it measures how many truly belong to the same ground-truth class.

High pairwise precision means the clustering makes few incorrect merges.

### Pairwise recall

Pairwise recall measures how well true same-class pairs are recovered.

It looks at all sample pairs that belong to the same ground-truth class. Among those pairs, it measures how many were placed in the same predicted cluster.

High pairwise recall means the clustering makes few incorrect splits.

### Pairwise F1

Pairwise F1 is the harmonic mean of pairwise precision and pairwise recall.

It balances two types of clustering errors:

- merging samples that should be separate, and
- splitting samples that should be together.

Pairwise F1 is often a better clustering-native alternative to classification F1.

### BCubed precision

BCubed precision measures cluster purity from each sample’s point of view.

For each sample, it looks at the predicted cluster containing that sample and checks what fraction of that cluster has the same true label as the sample. These per-sample precision values are averaged across all samples.

High BCubed precision means samples tend to be placed in clusters containing mostly members of their own true class.

### BCubed recall

BCubed recall measures class recovery from each sample’s point of view.

For each sample, it looks at the sample’s true class and checks what fraction of that class appears in the same predicted cluster as the sample. These per-sample recall values are averaged across all samples.

High BCubed recall means samples from the same true class tend to stay together.

### BCubed F1

BCubed F1 is the harmonic mean of BCubed precision and BCubed recall.

It balances sample-level cluster purity and sample-level class recovery.

BCubed F1 is useful when clusters have different sizes or when you want a metric that evaluates clustering quality from the perspective of individual samples.

## Requirements

```text
numpy
scipy
scikit-learn
```

## License

This project is licensed under the [MIT](https://github.com/semoglou/extclustval/blob/main/LICENSE) License.
