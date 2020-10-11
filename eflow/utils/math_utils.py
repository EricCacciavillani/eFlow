from math import log, e
import numpy as np
import math
import pandas as pd
from scipy import stats

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


def freedman_diaconis_bins(a):
    """
    Calculate number of hist bins using Freedman-Diaconis rule.
    From https://stats.stackexchange.com/questions/798/ and
    https://tinyurl.com/yxjqm7ff

    Args:
        a: np.array, pd.Series
            Continuous numerical data
    """
    a = np.asarray(a)
    if len(a) < 2:
        return 1
    h = 2 * stats.iqr(a) / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))


def auto_binning(a,
                 bins=None):
    """

        Takes a pandas series object and assigns generalized labels and binning
        dimensions.

    Args:
        df: pd.Dataframe
            Pandas Datafrane object

        df_features: DataFrameTypes from eflow
            DataFrameTypes object

        feature_name: string
            Name of the feature to extract the series from

        bins: int
            Number of bins to create.

    Returns:
        Gives back the bins and associated labels
    """

    if not bins:
        bins = 0

    if bins <= 0:
        bins = freedman_diaconis_bins(a)

    # Create bins of type pandas.Interval
    a = pd.Series(a).dropna()

    binned_list = list(pd.cut(a.sort_values(),
                              bins).unique())

    # Iterate through all possible bins
    bins = []
    labels = []
    for bin_count, binned_obj in enumerate(binned_list):

        # Extract from pandas.Interval into a list; just nicer to read
        binned_obj = [binned_obj.left, binned_obj.right]

        # -----
        if bin_count == 0:
            # Move bined value down so it properly captures the starting integer
            bins.append(binned_obj[0])
        labels.append(
            str(binned_obj[0]) + "+ " u"\u27f7 " + str(binned_obj[1]))

        bins.append(binned_obj[1])

    bins = [float(bins[i]) for i in range(0, len(bins))]
    return bins, labels