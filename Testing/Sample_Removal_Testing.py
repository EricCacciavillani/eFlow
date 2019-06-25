
# coding: utf-8

# In[1]:


import time
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
import copy
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.metrics import classification_report, accuracy_score
from subprocess import check_call
from sklearn.tree import export_graphviz
import operator
from dtreeviz.trees import *
from sklearn.datasets import load_iris
# from IPython import display

import sys
sys.path.append('..')

from Libraries.Utils.Global_Utils import *
from Libraries.SampleRemoval import *
from Libraries.ClusterMaster import *
from Libraries.DataFrameTypes import *


def testing_all_removal():
    
    iris = load_iris()

    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])


    sample_target_dict = dict()
    sample_target_dict["species"] = 2
    
    sample_remover = SampleRemoval(df=iris_df,
                                   sample_target_dict=sample_target_dict,
                                   columns_to_drop=list(sample_target_dict.keys()),
                                   show_visuals=False,
                                   pca_perc=.8)

        
    removal_index = sample_remover.remove_samples(new_sample_amount=38,
                                                  zscore_high=1.7,
                                                  annotate=True,
                                                  weighted_dist_value=1,
                                                  display_all_graphs=False,
                                                  create_visuals=False,
                                                  show_gif=False)
    
    print(removal_index)
    assert(removal_index == [119, 106, 101, 127, 100, 104, 120, 146, 142, 141, 103, 139])
    
    
def testing_similar_removal():
    
    iris = load_iris()

    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])


    sample_target_dict = dict()
    sample_target_dict["species"] = 2
    
    sample_remover = SampleRemoval(df=iris_df,
                                   sample_target_dict=sample_target_dict,
                                   columns_to_drop=list(sample_target_dict.keys()),
                                   pca_perc=.8,
                                   show_visuals=False)


    removal_index = sample_remover.remove_samples(new_sample_amount=40,
                                                  zscore_high=1.7,
                                                  annotate=True,
                                                  display_all_graphs=False,
                                                  remove_noise=False,
                                                  weighted_dist_value=1,
                                                  create_visuals=False,
                                                  show_gif=False)
    
    print(type(removal_index))
    print(removal_index)
    
    assert(removal_index == [101, 127, 100, 104, 120, 146, 142, 141, 103, 139])


def testing_similar_removal_with_weight():
    iris = load_iris()

    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])

    sample_target_dict = dict()
    sample_target_dict["species"] = 2

    sample_remover = SampleRemoval(df=iris_df,
                                   sample_target_dict=sample_target_dict,
                                   columns_to_drop=list(
                                       sample_target_dict.keys()),
                                   pca_perc=.8,
                                   show_visuals=False)

    removal_index = sample_remover.remove_samples(new_sample_amount=40,
                                                  zscore_high=1.7,
                                                  annotate=True,
                                                  display_all_graphs=False,
                                                  remove_noise=False,
                                                  weighted_dist_value=.2,
                                                  create_visuals=False,
                                                  show_gif=False)

    print(type(removal_index))
    print(removal_index)

    assert (removal_index == [101, 127, 100, 104, 120, 146, 142, 141, 103,
                              139])

def testing_random_partition_of_random_samples():

    assert(np.array_equal(
        random_partition_of_random_samples([0, 1, 2, 3, 4], 5, 6,
                                           random_state=4),
        np.array([[4, 2, 3, 0, 1],
                  [0, 1, 4, 3, 2],
                  [2, 4, 3, 0, 1],
                  [4, 2, 0, 3, 1],
                  [0, 2, 4, 3, 1]])))

def testing_kmeans_impurity_sample_removal():
    iris = load_iris()

    iris_df = pd.DataFrame(data=np.c_[iris['data'], iris['target']],
                           columns=iris['feature_names'] + ['species'])

    removal_index = kmeans_impurity_sample_removal(iris_df,
                                                   target="species",
                                                   majority_class=1,
                                                   majority_class_threshold=.5,
                                                   pca_perc=.85,
                                                   random_state=22)

    assert (removal_index == [75, 76, 77, 72, 83])