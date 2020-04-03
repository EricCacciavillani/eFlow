#!/usr/bin/env python
# coding: utf-8

# Find the markdown blocks that say interaction required! The notebook should take care of the rest!

# # Import libs

# In[1]:


import sys
import os
sys.path.append('..')
from eflow.foundation import DataPipeline,DataFrameTypes
from eflow.model_analysis import ClassificationAnalysis
from eflow.utils.modeling_utils import optimize_model_grid
from eflow.utils.eflow_utils import get_type_holder_from_pipeline, remove_unconnected_pipeline_segments
from eflow.utils.pandas_utils import data_types_table
from eflow.auto_modeler import AutoCluster
from eflow.data_pipeline_segments import DataEncoder

import pandas as pd
import numpy as np
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import copy
import pickle


# In[2]:


# # Additional add ons
# !pip install pandasgui
# !pip install pivottablejs
# clear_output()


# In[3]:


# ## Declare Project Variables

# ### Interaction required

# In[4]:


dataset_path = "Datasets/titanic_train.csv"

# -----
dataset_name = "Titanic Data"
pipeline_name = "Titanic Pipeline"

# -----


# -----
notebook_mode = False


# ## Clean out segment space

# In[5]:


remove_unconnected_pipeline_segments()


# # Import dataset

# In[6]:


df = pd.read_csv(dataset_path)
shape_df = pd.DataFrame.from_dict({'Rows': [df.shape[0]],
                                   'Columns': [df.shape[1]]})


# In[7]:


data_types_table(df)


# # Loading and init df_features

# In[8]:


# Option: 1
# df_features = get_type_holder_from_pipeline(pipeline_name)


# In[9]:


# Option: 2
df_features = DataFrameTypes()
df_features.init_on_json_file(os.getcwd() + f"/eflow Data/{dataset_name}/df_features.json")


# In[10]:


df_features.display_features(display_dataframes=True,
                             notebook_mode=notebook_mode)


# In[11]:


qualtative_features = df_features.string_features() | df_features.categorical_features()
qualtative_features


# # Any extra processing before eflow DataPipeline

# In[ ]:





# # Setup pipeline structure

# ### Interaction Required

# In[12]:


main_pipe = DataPipeline(pipeline_name,
                         df,
                         df_features)


# In[13]:


main_pipe.perform_pipeline(df,
                           df_features)


# In[14]:


df


# # Generate clustering models with automodeler

# In[15]:


auto_cluster = AutoCluster(df,
                           project_sub_dir=dataset_name,
                           overwrite_full_path=None,
                           notebook_mode=True,
                           pca_perc=.8)


# In[16]:


dp = [df.loc[0]]
print(dp)
dp = auto_cluster.apply_clustering_data_pipeline(dp)
print(dp)


# In[17]:


auto_cluster.get_scaled_data()[0]


# ### Temporialy remove dataframe to save RAM for processing

# In[18]:


del df


# # Inspect Hierarchical models

# In[19]:


# auto_cluster.visualize_hierarchical_clustering()


# In[20]:


# auto_cluster.create_elbow_models(sequences=10,
#                                  max_k_value=11,
#                                  display_visuals=True)


# In[21]:


df = pd.read_csv(dataset_path)
shape_df = pd.DataFrame.from_dict({'Rows': [df.shape[0]],
                                   'Columns': [df.shape[1]]})


# In[22]:


df_features.init_on_json_file(os.getcwd() + f"/eflow Data/{dataset_name}/df_features.json")
df_features.display_features(display_dataframes=True,
                             notebook_mode=notebook_mode)


# In[23]:


main_pipe.perform_pipeline(df,
                           df_features)


# In[24]:


df


# In[25]:


data_encoder = DataEncoder(create_file=False)


# In[26]:


data_encoder.revert_dummies(df,
                            df_features,
                            qualtative_features=list(df_features.get_dummy_encoded_features().keys()))


# In[27]:


df


# In[28]:


# auto_cluster.evaluate_all_models(df,
#                                  df_features)


# In[29]:


model = auto_cluster.get_all_cluster_models()["Somsc_Clusters=5"]


# In[30]:


def get_centers(data,
                model):
        try:
            return model.get_centers()
        except AttributeError:

            center_points = []

            for cluster_indexes in model.get_clusters():
                all_dps = np.matrix([data[i] for i in cluster_indexes])
                center_dp = all_dps.mean(0)

                # Grave Yard code: Use existing point rather than generating abstract average data point
                # np.absolute(all_dps - center_dp).sum(1).argmin()

                center_points.append(np.array(center_dp.tolist()[0]))

            return center_points

center_points = get_centers(auto_cluster.get_scaled_data(),
                            model)


# In[31]:


center_points


# In[32]:


len(center_points)


# In[33]:


dp


# In[34]:


from eflow.utils.modeling_utils import get_cluster_probas
get_cluster_probas(center_points,dp)


# In[35]:


from multiprocessing import Pool as ThreadPool
import math
import multiprocessing as mp
from functools import partial
from scipy import stats


# In[36]:


def euclidean_distance(v1, v2):
    dist = [(a - b)**2 for a, b in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


# In[37]:


def find_all_dist_with_target(data_points,
                              target_point):
    """
        Finds all distances between the target and the other points.
    """
    distances = []
    for dp in auto_cluster.get_scaled_data():
        distances.append(euclidean_distance(dp,target_point))

    return np.array(distances)


# In[38]:


# In[ ]:


import time
def multiprocess(func, jobs, cores):
    if cores == 1:
        logs = []
        for j in jobs:
            logs.append(func(j))

    elif cores == -1:
        with mp.Pool(mp.cpu_count()) as p:
            logs = list(p.map(func, jobs))

    elif cores > 1:
        with mp.Pool(cores) as p:
            logs = list(p.map(func, jobs))

    else:
        print('Error: jobs must be a positive integer')
        return False

for i in range(1,1000):
    print("Start")
    func = partial(find_all_dist_with_target,
                   auto_cluster.get_scaled_data())
    multiprocess(func,center_points,12)
    print(i)
    print()


# In[ ]:


import multiprocessing as mp
mp.cpu_count()


# In[ ]:
