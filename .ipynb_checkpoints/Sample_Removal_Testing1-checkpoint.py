
# coding: utf-8

# In[1]:


import pandas as pd
from dtreeviz.trees import *
from sklearn.datasets import load_iris
# from IPython import display

import sys
sys.path.append('..')

from eFlow.ClusterMaster import *
from eFlow.DataFrameTypes import *


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
                                                  show_visuals=False,
                                                  show_gif=True)
    
    assert(removal_index == [119, 106, 101, 127, 100, 104, 143, 146, 142, 141, 103, 139])
    
    
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
                                                  remove_noise=False,
                                                  weighted_dist_value=1,
                                                  show_gif=True,
                                                  show_visuals=False)
    
    assert(removal_index == [101, 127, 100, 104, 143, 146, 142, 141, 103, 139])

def testing_add_new_data_to_smaller():
    
    def bestest_func_ever():
        return [[2,3,52,5],[2,23,4,5],[21,13,4,95]]
    
    
    assert(bestest_func_ever() == [[26,3,52,5],[2,23,4,5],[21,13,4,95]])