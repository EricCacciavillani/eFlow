from math import log, e
import numpy as np
import math

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


def get_unbalanced_threshold(target_amount,
                             max_binary_threshold=.65):
    max_unbalanced_class_threshold = (1 / target_amount) / ((1/2)/(max_binary_threshold-(1/2))) + (
                1 / target_amount)
    min_unbalanced_class_threshold = (1 - max_unbalanced_class_threshold) / (
                target_amount - 1)

    return max_unbalanced_class_threshold, min_unbalanced_class_threshold


def calculate_entropy(labels,
                      base=None):
    n_labels = len(labels)

    if n_labels <= 1:
        return 0

    value, counts = np.unique(labels, return_counts=True)
    probs = counts / n_labels
    n_classes = np.count_nonzero(probs)

    if n_classes <= 1:
        return 0

    ent = 0.

    # Compute entropy
    base = e if base is None else base
    for i in probs:
        ent -= i * log(i, base)

    return ent


def euclidean_distance(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist