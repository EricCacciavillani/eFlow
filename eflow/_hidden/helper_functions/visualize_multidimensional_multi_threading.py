# Math utils
import os
from multiprocessing import Pool as ThreadPool
from functools import partial
import math
import numpy as np
import multiprocessing as mp
from collections import ChainMap

def find_all_dist_with_target(matrix,
                              index_array,
                              target_dp_index):
    """
        Finds all distances between the target and the other points.
    """

    distances = np.zeros(len(index_array) - target_dp_index - 1)
    for index, dp_index in enumerate(index_array[
                                     target_dp_index + 1:]):
        distances[index] = fast_eudis(matrix[target_dp_index],
                                      matrix[dp_index])
        # self.__pbar.update(1)

    all_distances_to_target = dict()

    all_distances_to_target[target_dp_index] = distances

    return all_distances_to_target


def find_all_distances_in_matrix(matrix):

    index_array = [i for i in range(0,
                                    len(matrix))]

    pool = ThreadPool(mp.cpu_count() - 2)

    func = partial(find_all_dist_with_target,
                   matrix,
                   index_array)

    all_dp_distances = list(
        pool.imap(func,
                  index_array[:-1]))

    all_dp_distances = dict(ChainMap(*all_dp_distances))

    pool.close()
    pool.join()

    return all_dp_distances


def weighted_eudis(v1,
                   v2,
                   feature_weights):
    dist = [((a - b) ** 2) * w for a, b, w in zip(v1, v2,
                                                  feature_weights)]
    dist = math.sqrt(sum(dist))
    return dist


def fast_eudis(v1,
               v2):
    dist = [((a - b) ** 2) for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist