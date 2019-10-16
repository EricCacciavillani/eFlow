from multiprocessing import Pool as ThreadPool

import numpy as np
import multiprocessing as mp
from functools import partial
import math

def find_all_dist_with_target(matrix,
                              index_array,
                              total_indexes,
                              feature_weights,
                              target_dp_index):
    """
        Finds all distances between the target and the other points.
    """
    distances = np.zeros(total_indexes - (target_dp_index + 1))
    for index, dp_index in enumerate(index_array[
                                     target_dp_index + 1:]):
        distances[index] = weighted_eudis(matrix[target_dp_index],
                                          matrix[dp_index],
                                          feature_weights)
        # self.__pbar.update(1)
    # shortest_dp_index = np.argmin(distances)

    all_distances_to_target = dict()

    all_distances_to_target[target_dp_index] = distances

    return all_distances_to_target


def find_all_distances_in_matrix(matrix,
                                 index_array,
                                 total_indexes,
                                 feature_weights):
    pool = ThreadPool(mp.cpu_count()-2)

    func = partial(find_all_dist_with_target,
                   matrix,
                   index_array,
                   total_indexes,
                   feature_weights)
    all_dp_distances = list(
        pool.imap_unordered(func,
                            index_array[:-1]))

    # Store dps relationships and the distances
    all_dp_dist_list = [np.array([])] * matrix.shape[0]

    # Convert map to list
    for dp_dict in all_dp_distances:
        all_dp_dist_list[list(dp_dict.keys())[0]] = \
            list(dp_dict.values())[0]

    return all_dp_dist_list


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