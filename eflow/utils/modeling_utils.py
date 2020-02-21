from sklearn.model_selection import GridSearchCV

from eflow.utils.math_utils import euclidean_distance

import numpy as np
from scipy import stats

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


def get_cluster_probas(center_points,
                       data):

    for dp in data:

        cluster_distances = []
        for cp in center_points:
            cluster_distances.append(euclidean_distance(cp, dp))

        probas = 1 - np.array(cluster_distances) / (
            np.array(cluster_distances).sum())
        return probas / probas.sum()


def find_all_zscore_distances_from_target(data_points,
                                          target_point):
    """
        Finds all distances between the target and the other points.
    """
    distances = []
    for dp in data_points:
        distances.append(euclidean_distance(dp, target_point))

    return stats.zscore(np.array(distances))


def optimize_model_grid(model,
                        X_train,
                        y_train,
                        param_grid,
                        scoring="f1_macro",
                        cv=5,
                        verbose=0,
                        n_jobs=1):
    """
    Desc:
        Finds the best parameters for a model; returns the model and parameters.

    Args:
        model:
            Machine learning model to fit across a cross fold with a cross

        X_train:
            Feature matrix.

        y_train:
            Target vector.

        param_grid:
            Dictionary with parameters names.

        scoring:
            String value to determine the metric to evaluate the best model.
            Link to all strings: http://tinyurl.com/y22f3m5k

        cv:
            Cross-validation strategy for 'training'.

        verbose:
            Controls the verbosity: the higher, the more messages.

        n_jobs:
            Number of jobs to run in parallel.

        Note:
            From the sklearn library;

    Returns:
        Finds the best parameters for a grid; returns the model and parameters.
        Link: http://tinyurl.com/y296u6mq
    """

    # Instantiate the GridSearchCV object
    model_cv = GridSearchCV(model,
                            param_grid,
                            cv=cv,
                            n_jobs=n_jobs,
                            verbose=verbose,
                            scoring=scoring)

    # Fit it to the data
    model_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print("Tuned Parameters: {}".format(model_cv.best_params_))
    print("Best score on trained data was {0:4f}".format(model_cv.best_score_))

    # Return model and parameters
    return model_cv.best_estimator_, model_cv.best_params_

