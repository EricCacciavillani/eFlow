
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
from Libraries.DataframeTypeHolder import *


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

def testing_random_partition_of_random_samples():
    
    import random
    def random_partition_of_random_samples(list_of_df_indexes,
                                           random_sampled_rows,
                                           random_sample_amount,
                                           random_state=None):
        
        # Convert to numpy array if list
        if isinstance(list_of_df_indexes, list):
            list_of_df_indexes = np.array(list_of_df_indexes)
        
        np.random.seed(random_state)
        for _ in range(np.random.randint(1,3)):
            np.random.shuffle(list_of_df_indexes)
        
        
        if random_sample_amount > len(list_of_df_indexes):
            random_sample_amount = len(list_of_df_indexes)
            
        return_matrix = np.zeros((random_sample_amount, random_sampled_rows))
        for i in range(random_sampled_rows):
            sub_list = list_of_df_indexes[:random_sample_amount]
            return_matrix[i] = sub_list
            np.random.shuffle(list_of_df_indexes)
        
        np.random.seed(None)
        return return_matrix
    
    
    assert(np.array_equal(
        random_partition_of_random_samples([0,1,2,3,4], 5, 6,random_state=4),
        np.array([[4, 2, 3, 0, 1],
                 [0, 1, 4, 3, 2],
                 [2, 4, 3, 0, 1],
                 [4, 2, 0, 3, 1],
                 [0, 2, 4, 3, 1]])))