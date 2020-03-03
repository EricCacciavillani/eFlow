
from eflow._hidden.parent_objects import AutoModeler
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow.utils.sys_utils import load_pickle_object,get_all_files_from_path, get_all_directories_from_path, pickle_object_to_file, create_dir_structure, write_object_text_to_file, json_file_to_dict, dict_to_json_file
from eflow.utils.eflow_utils import move_folder_to_eflow_garbage
from eflow.utils.modeling_utils import find_all_zscore_distances_from_target
from eflow.utils.math_utils import euclidean_distance
from eflow.data_analysis.feature_analysis import FeatureAnalysis
from eflow.data_pipeline_segments import DataEncoder

# Getting Sklearn Models
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram,set_link_color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Getting pyclustering
from pyclustering.cluster.kmeans import kmeans
from pyclustering.cluster.kmedians import kmedians
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.cluster.ema import ema
from pyclustering.cluster.cure import cure
from pyclustering.cluster.fcm import fcm
from pyclustering.cluster.somsc import somsc
from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer,random_center_initializer

# Visuals libs
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pylab as pl
import seaborn as sns
from IPython.display import display, HTML

# Misc
from collections import Counter
from scipy.stats import zscore
from kneed import KneeLocator
import pandas as pd
import six
import random
import numpy as np
import copy
import os
from tqdm import tqdm
import warnings


__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class AutoCluster(AutoModeler):
    """
        Analyzes the feature data of a pandas Dataframe object.
    """

    def __init__(self,
                 df,
                 project_sub_dir="",
                 project_name="Auto Clustering",
                 overwrite_full_path=None,
                 notebook_mode=False,
                 pca_perc=None):
        """
        Args:
            df: pd.Dataframe
                pd.Dataframe

            project_sub_dir: string
                Sub directory to write data.

            project_name: string
                Main project directory

            overwrite_full_path: string
                Overwrite full directory path to a given output folder

            notebook_mode: bool
                Display and show in notebook if set to true.
        """

        AutoModeler.__init__(self,
                             f'{project_sub_dir}/{project_name}',
                             overwrite_full_path)


        if os.path.exists(self.folder_path + "_Extras"):
            move_folder_to_eflow_garbage(self.folder_path+"_Extras",
                                         "Auto Clustering")


        # Define model
        self.__cluster_models_paths = dict()

        self.__notebook_mode = copy.deepcopy(notebook_mode)

        self.__models_suggested_clusters = dict()

        self.__pca = None
        self.__pca_perc = None

        self.__first_scaler = None
        self.__second_scaler = None
        self.__cutoff_index = None
        self.__pca_perc = pca_perc

        # --- Apply pca ---
        if pca_perc:

            # Create scaler object
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df)

            self.__first_scaler = copy.deepcopy(scaler)

            print("\nInspecting scaled results!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=df.columns,
                                          sub_dir="PCA",
                                          filename="Applied scaler results")

            pca, scaled = self.__visualize_pca_variance(scaled)

            self.__pca = pca

            # Generate "dummy" feature names
            pca_feature_names = ["PCA_Feature_" +
                                 str(i) for i in range(1,
                                                       len(df.columns) + 1)]

            print("\nInspecting applied scaler and pca results!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=pca_feature_names,
                                          sub_dir="PCA",
                                          filename="Applied scaler and PCA results")

            if pca_perc < 1.0:
                # Find cut off point on cumulative sum
                cutoff_index = np.where(
                    pca.explained_variance_ratio_.cumsum() > pca_perc)[0][0]
            else:
                cutoff_index = scaled.shape[1] - 1

            print(
                "After applying pca with a cutoff percentage {0}%"
                " for the cumulative index. Using features 1 to {1}".format(
                    pca_perc, cutoff_index + 1))

            print("Old shape {0}".format(scaled.shape))

            scaled = scaled[:, :cutoff_index + 1]
            pca_feature_names = pca_feature_names[0: cutoff_index + 1]

            print("New shape {0}".format(scaled.shape))

            scaled = scaler.fit_transform(scaled)

            print("\nInspecting data after final scaler applied!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=pca_feature_names,
                                          sub_dir="PCA",
                                          filename="Applied final sclaer to process.")

            self.__second_scaler = copy.deepcopy(scaler)

            self.__scaled = scaled
            self.__cutoff_index = cutoff_index

        # Assumed PCA has already been applied; pass as matrix
        else:
            self.__scaled = df.values

        if os.path.exists(self.folder_path + "Models"):

            print("Found past models in directory structure! Attempting to re-initalize models...")

            cluster_directories = get_all_directories_from_path(self.folder_path + "Models")
            for cluster_dir in cluster_directories:
                if cluster_dir[0] != ".":
                    for dir in get_all_directories_from_path(self.folder_path + f"Models/{cluster_dir}"):
                        if dir[0] != ".":
                            for model_name in get_all_files_from_path(self.folder_path + f"Models/{cluster_dir}/{dir}",
                                                                      ".pkl"):

                                model_path = self.folder_path + f"Models/{cluster_dir}/{dir}/{model_name}"

                                try:
                                    obj = load_pickle_object(model_path)

                                    if isinstance(obj,list) or isinstance(obj, np.ndarray):
                                        continue
                                    else:
                                        del obj

                                    model_name = model_name.split(".")[0]

                                    self.__cluster_models_paths[
                                        model_name] = model_path
                                except Exception:
                                    pass

            if self.__cluster_models_paths:
                for model_name, model_path in self.__cluster_models_paths.items():
                    print(f"{model_name} was found at {model_path}\n")

        # Save objects to directory structure
        if self.__pca:
            pipeline_path = create_dir_structure(self.folder_path,
                                                 "Data Pipeline")


            # Pickle data pipeline objects
            pickle_object_to_file(self.__pca,
                                  pipeline_path,
                                  "PCA")

            pickle_object_to_file(self.__first_scaler,
                                  pipeline_path,
                                  "First Scaler")

            pickle_object_to_file(self.__second_scaler,
                                  pipeline_path,
                                  "First Scaler")

            pickle_object_to_file(self.__pca_perc,
                                  pipeline_path,
                                  "PCA Percentage")

            # Save Dimensions and Cutoff Index
            write_object_text_to_file(self.__cutoff_index,
                                      pipeline_path,
                                      "Cutoff Index")

            write_object_text_to_file(self.__cutoff_index + 1,
                                      pipeline_path,
                                      "Dimensions")



    # --- Getters/Setters
    def get_scaled_data(self):
        """
        Desc:
            Get a copy of the stored data

        Returns:
            Returns the stored data
        """
        return copy.deepcopy(self.__scaled)

    def get_all_cluster_models(self):
        """
        Desc:
            Gets the model names and model instances in dictionary form.

        Return:
            Returns the model name to model instance dict
        """
        tmp_dict = dict()
        for model_name, model_path in self.__cluster_models_paths.items():

            try:

                model = load_pickle_object(model_path)

                tmp_dict[model_name] = model
            except EOFError:
                pass

        return tmp_dict

    def delete_scaled_data(self):
        """
        Desc:
            Removes the matrix data in order to save RAM when running
            analysis on the actual data.
        """
        del self.__scaled
        self.__scaled = None

    def apply_clustering_data_pipeline(self,
                                       data):
        """
        Desc:
            Apply the scaler, dimension reduction transformation, matrix shrink
            and second scaler to the data.

        Args:
            data: np.matrix, list of lists, pd.DataFrame
                Data that is similar in form and value structure to the data passed
                on initialization.

        Returns:
            Returns back data after transformations are applied.
        """

        data = self.__first_scaler.transform(data)
        data = self.__pca.transform(data)
        data = data[:, :self.__cutoff_index + 1]
        data = self.__second_scaler.transform(data)

        return data

    def get_model(self,
                  model_name):
        """
        Desc:
            Get the model object from the stored dirctories based on the model
            name.

        Args:
            model_name: str
                Name of the model that references the saved model in
                the provided directory.

        Returns:
            Gives back the model object.
        """

        return load_pickle_object(self.__cluster_models_paths[model_name])


    def visualize_hierarchical_clustering(self,
                                          linkage_methods=None,
                                          display_print=True,
                                          display_visuals=True):
        """
        Desc:
            Displays hierarchical cluster graphs with provided methods.

        Args:
            linkage_methods: list of strings
                All methods applied to the linkage

            display_print: bool
                Display print outputs

            display_visuals: bool
                Display plot data if set to ture.

        Returns:
            All cluster counts found.
        """

        best_clusters = []

        # ---
        if not linkage_methods:
            linkage_methods = ["complete",
                               "single",
                               "weighted",
                               "ward",
                               "average",
                               "centroid",
                               "median"]

        # Apply methods to each dendrogram
        for method in linkage_methods:

            if display_print:
                print(f"Creating graphic for Hierarchical Clustering Method: {method}...")

            # Create mergings
            mergings = linkage(self.__scaled, method=method)

             # {"Yellow":"#d3d255",
             # "Magenta":"#c82bc9",
             # "Black":"#030303",
             # "Red":"#ff403e",
             # "Green":"#3f9f3f",
             # "Cyan":"#0ec1c2",
             # "Brown": "#775549",
             # "Silver": "#C0C0C0",
             # "Blue":"#24326f",
             # "Orange":"#cc7722",
             # "Mauve":"#9c7c8c"}

            # Set plot
            plt.figure(figsize=(12, 7))
            set_link_color_palette(None)

            # Plot the dendrogram, using varieties as labels
            color_list = dendrogram(mergings,
                                    labels=list(range(0, len(self.__scaled,))),
                                    leaf_rotation=90,
                                    leaf_font_size=3)["color_list"]

            method = method.capitalize()
            plt.title(f"Hierarchical Clustering Method : \'{method}\'")

            # -----
            self.save_plot("Hierarchical Clustering",
                           f"Hierarchical Clustering Method {method} without legend")

            del mergings


            # Create proper cluster names based on the color names
            color_cluster_count = dict()
            last_color = None
            known_colors = set()
            color_cluster_order = list()
            seq_len = 0
            i = 0

            # -----
            for color in copy.deepcopy(color_list):

                # Proper color name
                color = self.__get_color_name(color)

                # Name for old cluster color sequence found
                if color in known_colors:
                    color_list[i] = f"{color} cluster {color_cluster_count[color]}"

                # Name for new cluster color sequence found
                else:
                    color_list[i] = f"{color} cluster 0"

                # Track unique cluster color order
                if color_list[i] not in color_cluster_order:
                    color_cluster_order.append(color_list[i])

                if last_color:
                    # Sequence of color has yet to be broken
                    if last_color == color:

                        # Only really need to check if the sequence has a length of 1
                        if seq_len <= 2:
                            seq_len += 1

                    # Sequence break
                    else:

                        # Valid cluster found
                        if seq_len > 1:

                            # Track all known color names
                            if last_color not in known_colors:
                                known_colors.add(last_color)

                            # If color is repeated; then make a new cluster count name
                            if last_color not in color_cluster_count.keys():
                                color_cluster_count[last_color] = 1
                            else:
                                color_cluster_count[last_color] += 1

                        # Invalid color cluster found; apply needed changes
                        else:
                            color_list.pop(i-1)
                            i -= 1

                        seq_len = 0

                last_color = color
                i += 1

            # Create legend for each cluster name and the amount of per sample
            counter_object = Counter(color_list)
            cluster_color_count = dict()
            handles = []

            # Make sure the legend is in the same order the cluster appear in the dendrogram
            for color_cluster_name in color_cluster_order:
                if color_cluster_name in counter_object.keys():
                    cluster_color_count[color_cluster_name] = counter_object[color_cluster_name]
                    try:
                        handles.append(mpatches.Patch(
                            color=color_cluster_name.split(" cluster ")[0],
                            label=color_cluster_name + f": {counter_object[color_cluster_name]} samples"))

                    except:
                        handles.append(mpatches.Patch(
                            color="black",
                            label=color_cluster_name + f": {counter_object[color_cluster_name]} samples"))

            # Plot the legend and save the plot
            plt.legend(handles=handles,
                       loc='upper right',
                       bbox_to_anchor=(1.32, 1.01),
                       title=f"Clusters ({len(handles)})")

            best_clusters.append(len(handles))
            if display_visuals and self.__notebook_mode:
                plt.show()

            self.save_plot("Hierarchical Clustering",
                           f"Hierarchical Clustering Method {method} with legend")

            plt.close('all')

        # Save results into _Extras folder
        best_clusters.sort()
        self.__models_suggested_clusters["Hierarchical Clustering"] = best_clusters
        self.__save_update_best_model_clusters()
        return best_clusters


    # def __visualize_clusters(self, model, output_path, model_name=""):
    #     """
    #         Creates visualization of clustering model on given data.
    #     """
    #     markers = ["+", "*", "X", "o", "v", "P", "H", "4", "p", "D", "s",
    #                "1", "x", "d", "_"]
    #     colors = ['b', 'g', 'r', 'c', 'm', 'y',
    #               '#007BA7', '#ff69b4', '#CD5C5C', '#7eab19', '#1a4572',
    #               '#2F4F4F', '#4B0082', '#d11141', '#5b2504']
    #
    #     # Display ranking on color based on amount data points per cluster
    #     unique, counts = np.unique(model.labels_, return_counts=True)
    #     cluster_names = ["Cluster:" + str(cluster_label)
    #                      for cluster_label in unique]
    #     self.__display_rank_graph(feature_names=cluster_names,
    #                               metric=counts,
    #                               title=model_name,
    #                               output_path=output_path,
    #                               model_name=model_name,
    #                               y_title="Clusters",
    #                               x_title="Found per cluster")
    #     pl.figure(figsize=(8, 7))
    #
    #     # Display clustered graph
    #     cluster_array = list(range(0, len(cluster_names)))
    #     scaled_cluster_label = np.hstack(
    #         (self.__scaled, np.reshape(
    #             model.labels_.astype(int), (self.__scaled.shape[0], 1))))
    #     for i in range(0, scaled_cluster_label.shape[0]):
    #         cluster_label = int(scaled_cluster_label[i][-1])
    #         cluster_array[cluster_label] = pl.scatter(
    #             scaled_cluster_label[i, 0], scaled_cluster_label[i, 1],
    #             c=colors[cluster_label], marker=str(markers[cluster_label]))
    #
    #     pl.legend(cluster_array, cluster_names)
    #     pl.title(model_name + ' visualized with data', fontsize=15)
    #     self.__image_processing_utils(output_path,
    #                           model_name + "_Visualized_Cluster")
    #     plt.show()
    #     plt.close()
    #     pl.close()
    def create_elbow_models(self,
                            model_names=["K-Means",
                                         "K-Medians",
                                         "K-Medoids",
                                         "Somsc",
                                         "Cure",
                                         "Fuzzy C-Means"],
                            sequences=3,
                            max_k_value=15,
                            display_visuals=True):
        """
        Desc:
            Create multiple sequences of defined clusters and their related
            inertia values to find the most agreed

        Args:
            model_names: list of strings
                Defines which models should be created.

            sequences: int
                How many model sequences to create

            max_k_value: int
                Max clusters to create

            display_visuals: bool
                Display graphics and tables if set to True.
        """

        model_names = set(model_names)

        # Model names and their model instances
        names_model_dict = {"K-Means":kmeans,
                            "K-Medians":kmedians,
                            "K-Medoids":kmedoids,
                            "Somsc":somsc,
                            "Cure":cure,
                            "Fuzzy C-Means": fcm}

        # Iterate through passed model names
        for name in model_names:

            if name in names_model_dict.keys():

                # Only requires 1 elbow sequence needed
                if name == "Somsc" or name == "Cure":
                    best_clusters = self.__create_elbow_seq(name,
                                                            names_model_dict[name],
                                                            sequences=1,
                                                            max_k_value=max_k_value,
                                                            display_visuals=display_visuals)
                else:
                    best_clusters = self.__create_elbow_seq(name,
                                                            names_model_dict[name],
                                                            sequences=sequences,
                                                            max_k_value=max_k_value,
                                                            display_visuals=display_visuals)

                # Save cluster results in
                best_clusters.sort()
                self.__models_suggested_clusters[name] = best_clusters
                self.__save_update_best_model_clusters()
            else:
                raise UnsatisfiedRequirments(f"Unknown model name passed: \"{name}\"")

        return best_clusters

    def create_agglomerative_models(self,
                                    n_cluster_list,
                                    linkage_methods=None):
        """
        Desc:
            Create multiple agglomerative models based on a list of
            'n_clusters' values and defined linkage methods.
        """
        raise ValueError("This functionality has been clamped!...Hopefully someone get's this refrence")
        pass
        # if isinstance(n_cluster_list, int):
        #     n_cluster_list = [n_cluster_list]
        #
        # if not linkage_methods:
        #     linkage_methods = ["ward", "complete", "average", "single"]
        #
        # knn_graph = kneighbors_graph(
        #     self.__scaled, len(
        #         self.__scaled) - 1, include_self=False)
        #
        # for n_clusters in n_cluster_list:
        #     for connectivity in (None, knn_graph):
        #
        #         for _, linkage in enumerate(linkage_methods):
        #             model = AgglomerativeClustering(linkage=linkage,
        #                                             connectivity=connectivity,
        #                                             n_clusters=n_clusters)
        #             model.fit(self.__scaled)
        #             self.__cluster_models[
        #                 "AgglomerativeClustering_{0}_"
        #                 "cluster{1}_Connectivity{2}".format(
        #                     linkage,
        #                     n_clusters, connectivity is not None)] = model
        #
        #             print(
        #                 "Successfully generate Agglomerative model with "
        #                 "linkage {0} on n_clusters={1}".format(
        #                     linkage, n_clusters))

        # self.__models_suggested_clusters["Agglomerative models"] =

    def evaluate_all_models(self,
                            df,
                            df_features,
                            qualitative_features=[],
                            zscore=None):
        """
        Desc:
            Loop through all models and evaluate the given model with
            'evaluate_model'. Read 'evaluate_model' to learn more.

        Args:
            df: pd.DataFrame
                Pandas dataframe

            df_features: DataFrameTypes object from eflow.
                DataFrameTypes object; organizes feature types into groups.

            zscore: float, int, None
                Will find the distances between given the given clusters then
                we apply data pipeline.
        """

        for model_name, model_path in self.__cluster_models_paths.items():
            self.evaluate_model(model_name=model_name,
                                df=df,
                                df_features=df_features,
                                qualitative_features=qualitative_features,
                                zscore=zscore)
            print("------" * 10)

    def evaluate_model(self,
                       model_name,
                       df,
                       df_features,
                       qualitative_features=[],
                       zscore=None,
                       target_features=None,
                       display_visuals=False,
                       display_print=False,
                       suppress_runtime_errors=True,
                       aggregate_target_feature=True,
                       selected_features=None,
                       extra_tables=True,
                       statistical_analysis_on_aggregates=True):
        """
        Desc:
            The main purpose of 'evaluate_model' is to display/save tables/plots
            accoiated with describing the model's 'findings' for each cluster.

        Args:

            model_name: str
                The string key to give the dict

            df: pd.Dataframe
                Dataframe object

            df_features: DataFrameTypes object from eflow.
                DataFrameTypes object; organizes feature types into groups.

            qualitative_features: list
                Any features that are currently dummy encoded will be reversed
                when doing feature analysis.

            zscore: float, int, None, or list of ints/floats
                After calculating each distance to the center point; a zscore
                value for distance will be found. Setting this parameter defines
                a cut off point where data exceeding a certain zscore will be
                ignored when doing FeatureAnalysis

            target_features: collection of strings or None
                A feature name that both exists in the init df_features
                and the passed dataframe.

                Note
                    If init to 'None' then df_features will try to extract out
                    the target feature.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            extra_tables: bool
                When handling two types of features if set to true this will
                    generate any extra tables that might be helpful.
                    Note -
                        These graphics may create duplicates if you already applied
                        an aggregation in 'perform_analysis'

            aggregate_target_feature: bool
                Aggregate the data of the target feature if the data is
                non-continuous data.

                Note
                    In the future I will have this also working with continuous
                    data.

            selected_features: collection object of features
                Will only focus on these selected feature's and will ignore
                the other given features.

            statistical_analysis_on_aggregates: bool
                If set to true then the function 'statistical_analysis_on_aggregates'
                will run; which aggregates the data of the target feature either
                by discrete values or by binning/labeling continuous data.
        """

        # Get/Set model related data
        model = load_pickle_object(self.__cluster_models_paths[model_name])
        model_path = self.__cluster_models_paths[model_name]
        model_dir = os.path.dirname(model_path)
        model_sub_dir = model_dir.replace(self.folder_path,"",1)
        center_points = load_pickle_object(os.path.dirname(
            self.__cluster_models_paths[model_name]) + "/Center points.pkl")
        cluster_labels = json_file_to_dict(f"{model_dir}/Cluster Labels.json")

        # Store single instance of int/float/None in a list
        if isinstance(zscore,float) or isinstance(zscore,type(None)) or isinstance(zscore,int):
            zscore = [zscore]

        # Iterate through all zscore values
        for z_val in zscore:

            cluster_labels_lengths = dict()

            # Handling all zscore values
            no_zscore = False

            if not z_val:
                z_val = float("inf")
                no_zscore = True
            else:
                z_val = round(z_val, 5)

            print(f"Model Name: {model_name}")
            print(f"Clusters: {len(model.get_clusters())}")
            print(f"Distance Zscore: {z_val}")
            print()

            for i,cluster_indexes in enumerate(model.get_clusters()):

                label = cluster_labels[str(i)]

                print(f"Feature Analysis on cluster {label}")

                # Create cluster dataframe
                tmp_df = df.loc[cluster_indexes].reset_index(drop=True)

                # Find and apply zscore cut off point
                bool_array = find_all_zscore_distances_from_target(self.apply_clustering_data_pipeline(tmp_df.values),
                                                                   center_points[i])

                bool_array = (bool_array >= -z_val) & (bool_array <= z_val)

                removed_dp_count = len(bool_array) - bool_array.sum()
                removed_dp_percentage = removed_dp_count / len(tmp_df)

                tmp_df = tmp_df[bool_array].reset_index(drop=True)

                del bool_array

                # Decode and revert dummies data
                tmp_df_features = copy.deepcopy(df_features)

                data_encoder = DataEncoder(create_file=False)

                data_encoder.revert_dummies(tmp_df,
                                            tmp_df_features,
                                            qualitative_features=qualitative_features)

                data_encoder.decode_data(tmp_df,
                                         tmp_df_features,
                                         apply_value_representation=False)

                data_encoder.apply_value_representation(tmp_df,
                                                        df_features)
                del data_encoder

                cluster_labels_lengths[label] = len(tmp_df)

                # Handle naming/managing the directory structure
                if no_zscore:
                    create_dir_structure(directory_path=model_dir,
                                         create_sub_dir="No Distance Zscore")

                    self.__create_cluster_profile(tmp_df,
                                                  tmp_df_features,
                                                  sub_dir=f"{model_sub_dir}/No Distance Zscore/Cluster: {label}")


                    feature_analysis = FeatureAnalysis(tmp_df_features,
                                                       overwrite_full_path=f"{model_dir}/No Distance Zscore",
                                                       notebook_mode=self.__notebook_mode)
                else:
                    create_dir_structure(directory_path=model_dir,
                                         create_sub_dir=f"Distance Zscore = {z_val}/Cluster: {label}")

                    self.__create_cluster_profile(tmp_df,
                                                  tmp_df_features,
                                                  sub_dir=f"{model_sub_dir}/Distance Zscore = {z_val}/Cluster: {label}")


                    feature_analysis = FeatureAnalysis(tmp_df_features,
                                                       overwrite_full_path=f"{model_dir}/Distance Zscore = {z_val}",
                                                       notebook_mode=self.__notebook_mode)

                # Analyze the cluster dataframe
                feature_analysis.perform_analysis(tmp_df,
                                                  dataset_name=f"Cluster: {label}",
                                                  target_features=target_features,
                                                  display_visuals=display_visuals,
                                                  display_print=display_print,
                                                  dataframe_snapshot=True,
                                                  save_file=True,
                                                  suppress_runtime_errors=suppress_runtime_errors,
                                                  aggregate_target_feature=aggregate_target_feature,
                                                  selected_features=selected_features,
                                                  extra_tables=extra_tables,
                                                  statistical_analysis_on_aggregates=statistical_analysis_on_aggregates)

                # Save removed data points
                write_object_text_to_file(removed_dp_count,
                                          f"{feature_analysis.folder_path}/Cluster: {label}",
                                          filename="Removed dp count")

                write_object_text_to_file(removed_dp_percentage,
                                          f"{feature_analysis.folder_path}/Cluster: {label}",
                                          filename="Removed dp percentage")

            if no_zscore:
                self.__display_cluster_label_rank_graph(list(cluster_labels_lengths.keys()),
                                                        list(cluster_labels_lengths.values()),
                                                        sub_dir=f"{model_dir}/No Distance Zscore",
                                                        display_visuals=self.__notebook_mode)
            else:
                self.__display_cluster_label_rank_graph(list(cluster_labels_lengths.keys()),
                                                        list(cluster_labels_lengths.values()),
                                                        sub_dir=f"{model_sub_dir}/Distance Zscore = {z_val}",
                                                        display_visuals=self.__notebook_mode)

            print("###" * 4)


    def __create_cluster_profile(self,
                                 df,
                                 df_features,
                                 sub_dir):
        """
        Desc:
            Generate a cluster profile based on the clustered data. A cluster
            profile gets the mean of a numerical series data and the mode of
            a non-numerical one.

        Args:
            df: pd.DataFrame
                Dataframe object

            df_features: DataFrameTypes object from eflow.
                DataFrameTypes object; organizes feature types into groups.

            sub_dir: string
                Sub directory to create when writing data.
        """
        cluster_profile_dict = dict()

        for feature_name in df.columns:

            if feature_name not in df_features.all_features():
                raise ValueError(f"No feature named {feature_name} was found in df_features.")

            if feature_name in df_features.continuous_numerical_features():
                cluster_profile_dict[feature_name] = df[feature_name].mean()
            else:
                cluster_profile_dict[feature_name] = ":".join([str(i) for i in df[feature_name].mode()])
                if feature_name in df_features.bool_features():
                    if cluster_profile_dict[feature_name] == "1":
                        cluster_profile_dict[feature_name] = True
                    elif cluster_profile_dict[feature_name] == "0":
                        cluster_profile_dict[feature_name] = False

        self.save_table_as_plot(pd.DataFrame(cluster_profile_dict, index=[0]).transpose(),
                                sub_dir=sub_dir,
                                filename="Cluster Profile",
                                show_index=True)

    def __inspect_feature_matrix(self,
                                 matrix,
                                 feature_names,
                                 sub_dir,
                                 filename):
        """
        Desc:
            Creates a dataframe to quickly analyze a matrix of data by mean and
            standard deviation.

        Args:

            matrix: list of lists, np.matrix, pd.DataFrame
                Matrix data to convert to mean and

            feature_names: list of strings
                Each dimension's name.

            sub_dir: string
                Sub directory to create when writing data.

            filename: string
                Name of the file
        """

        # Calculate mean and std for each dimension
        mean_matrix = np.mean(matrix, axis=0)
        std_matrix = np.std(matrix, axis=0)

        # Create relationship dict
        data_dict = dict()
        for index, feature_name in enumerate(feature_names):
            data_dict[feature_name] = [mean_matrix[index],
                                       std_matrix[index]]

        # Convert to dataframe, display, and save to directory.
        tmp_df = pd.DataFrame.from_dict(data_dict,
                                        orient='index',
                                        columns=['Mean', 'Standard Dev'])

        if self.__notebook_mode:
            display(tmp_df)
        else:
            print(tmp_df)

        self.save_table_as_plot(tmp_df,
                                sub_dir=sub_dir,
                                filename=filename,
                                show_index=True,
                                format_float_pos=5)


    def __visualize_pca_variance(self,
                                 data):
        """
        Desc:
            Visualize PCA matrix feature importance.

        Args:
            data: list of list, np.matrix,
                Values to have pca applied too.

        Credit to favorite teacher Narine Hall for making this function.
        I wouldn't be the programmer I am today if it wasn't for her.
        """

        # Check for pca variance
        pca = PCA()
        data = pca.fit_transform(data)

        # ----
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_)
        plt.xticks()
        plt.ylabel('Variance ratio')
        plt.xlabel('PCA feature')
        plt.tight_layout()

        self.save_plot("PCA",
                       "PCA Feature Variance Ratio")

        if self.__notebook_mode:
            plt.show()
            plt.close("all")

        # ----
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_.cumsum())
        plt.xticks()
        plt.ylabel('Cumulative sum of variances')
        plt.xlabel('PCA feature')
        plt.tight_layout()
        self.save_plot("PCA",
                       "PCA Cumulative Sum of Variances")

        if self.__notebook_mode:
            plt.show()
            plt.close("all")

        return pca, data


    def __create_elbow_seq(self,
                           model_name,
                           model_instance,
                           sequences,
                           max_k_value,
                           display_visuals):
        """
        Desc:
            Fit's multiple cluster models and calculates each model's intertia
            value. Because the clustering is subjective; finding the 'elbow' of
            the elbow determines the 'best' model. That job is passed on to
            "__find_best_elbow_models" to find the elbow of each sequence and
            then the best matching elbow of all elbows.

        Args:
            model_name: string
                Model's name

            model_instance: pyclustering model
                Model's instance

            sequences: int
                How many model sequences to create

            max_k_value: int
                How long to create the model sequence

            display_visuals: bool
                Display the graphics

        Returns:
            Get's the best cluster amounts
        """

        max_k_value += 1

        # Matrix declarations
        k_models = []
        inertias = []

        for elbow_seq_count in range(0,sequences):
            tmp_inertias = []
            tmp_k_models = []

            # Set up progress bar
            if display_visuals:
                pbar = tqdm(range(1,max_k_value), desc=f"{model_name} Elbow Seq Count {elbow_seq_count + 1}")
            else:
                pbar = range(1,max_k_value)

            # Find random center points for clusters
            initial_centers = self.__create_random_initial_centers(model_name,
                                                                   max_k_value)
            for k_val in pbar:

                if display_visuals:
                    pbar.set_postfix(model_count=k_val,
                                     refresh=True)

                model = self.__create_pyclustering_model(model_name=model_name,
                                                         model_instance=model_instance,
                                                         initial_centers=initial_centers,
                                                         k_val=k_val)

                # Run cluster analysis and obtain results.
                model.process()
                final_centers = np.array(self.__get_centers(model))
                labels = [self.__nearest_cluster(final_centers, dp)
                          for dp in self.__scaled]

                inertia = sum(((final_centers[l] - x) ** 2).sum()
                              for x, l in zip(self.__scaled, labels))

                # Append the inertia to the list of inertias
                tmp_inertias.append(inertia)
                tmp_k_models.append(model)

            k_models.append(tmp_k_models)
            inertias.append(tmp_inertias)

        return self.__find_best_elbow_models(model_name,
                                             k_models,
                                             inertias,
                                             display_visuals)

    def __get_centers(self,
                      model):
        """
        Desc:
            Get's/creates the center point of each cluster.
        Args:
            model: pyclustering
                Clustering model instance.

        Returns:
            The center points of the clusters
        """
        try:
            return model.get_centers()
        except AttributeError:

            center_points = []

            for cluster_indexes in model.get_clusters():
                all_dps = np.matrix([self.__scaled[i] for i in cluster_indexes])
                center_dp = all_dps.mean(0)

                # Grave Yard code: Use existing point rather than generating abstract average data point
                # np.absolute(all_dps - center_dp).sum(1).argmin()

                center_points.append(np.array(center_dp.tolist()[0]))

            return center_points


    def __create_pyclustering_model(self,
                                    model_name,
                                    model_instance,
                                    initial_centers,
                                    k_val):
        """
        Desc:
            Generates simple clustering model's that only require the clustering
            amount.

        Args:
            model_name: string
                Name of the model

            model_instance: pyclustering
                Clustering model

            initial_centers: list of list of floats
                Center points of each cluster

            k_val:
                Value of how many cluster's that should be generated.

        Returns:
            Returns a init instance of the needed pyclustering model.
        """


        if model_name == "Somsc" or model_name == "Cure":
            model = model_instance(self.__scaled,
                                   k_val)
        else:
            model = model_instance(self.__scaled,
                                   initial_centers[0:k_val])
        return model

    def __create_random_initial_centers(self,
                                        model_name,
                                        k_val):
        """
        Desc:
            Generates multiple random starting center points.

        Args:
            model_name: string
                Name of the given model (some models have different requirements)

            k_val: int
                How many random center points/clusters need to be created.
        """
        if model_name == "Somsc" or model_name == "Cure":
            return None
        elif model_name == "K-Medoids":
            return [i for i in self.__get_unique_random_indexes(k_val)]

        elif model_name == "K-Means" or model_name == "Fuzzy C-means":
            return kmeans_plusplus_initializer(self.__scaled,
                                               k_val).initialize()
        else:
            return random_center_initializer(self.__scaled,
                                             k_val).initialize()

    def __find_best_elbow_models(self,
                                 model_name,
                                 k_models,
                                 inertias,
                                 display_visuals=True):
        """
        Desc:
            Find the elbow of each sequence and then find the best matching
            elbow that all sequences agree upon.

        Args:
            model_name: string
                Name of all model instance

            k_models: list of list of pyclustering models
                All pyclustering models

            inertias: list of list of floats
                All model's intertia's

            display_visuals: bool
                Determines if graphics should be displayed

        Returns:
            Returns back the best numbers of clusters.
        """

        ks = range(1,
                   len(inertias[0]) + 1)

        # Set graphic info
        plt.figure(figsize=(13, 6))
        plt.title(f"All possible {model_name} Elbow's", fontsize=15)
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)

        # ----
        elbow_inertias_matrix = None
        inertias_matrix = None
        elbow_models = []
        elbow_sections = []
        center_elbow_count = dict()
        proximity_elbow_count = dict()
        elbow_cluster = None

        # Plot inertias against ks (y values against x values)
        for i in range(0,len(inertias)):

            # Determine curve
            curve = self.__determine_curve(inertias[i])

            if curve == "concave":
                online = True
            else:
                online = False

            # Find the best cluster model
            elbow_cluster = KneeLocator(ks,
                                        inertias[i],
                                        curve="convex",
                                        online=online,
                                        direction='decreasing').elbow
            # print(f"elbow_cluster:{elbow_cluster}")

            if elbow_cluster == 1 or not elbow_cluster:
                print("Elbow was either one or None for the elbow seq.")
                continue

            # Plot sequence
            plt.plot(ks,
                     inertias[i],
                     '-o',
                     color='#367588',
                     alpha=0.5)

            # Count all center elbows found
            if str(elbow_cluster) not in center_elbow_count.keys():
                center_elbow_count[str(elbow_cluster)] = 1
            else:
                center_elbow_count[str(elbow_cluster)] += 1

            # Get elbow section (1 before center and 1 after center)
            for k_val in [elbow_cluster - 1, elbow_cluster, elbow_cluster + 1]:
                elbow_sections.append([ks[k_val - 1],inertias[i][k_val - 1]])

                if str(k_val) not in proximity_elbow_count.keys():
                    proximity_elbow_count[str(k_val)] = 1
                else:
                    proximity_elbow_count[str(k_val)] += 1

            # Get inertia values
            if isinstance(elbow_inertias_matrix, type(None)):
                inertias_matrix = np.matrix([inertias[i]])
                elbow_inertias_matrix = np.matrix(inertias[i][elbow_cluster - 2:elbow_cluster + 1])
            else:
                inertias_matrix = np.vstack([inertias_matrix, inertias[i]])

                elbow_inertias_matrix = np.vstack(
                    [elbow_inertias_matrix, inertias[i][elbow_cluster - 2:elbow_cluster + 1]])

            elbow_models.append(k_models[i][elbow_cluster - 2:elbow_cluster + 1])

        # Plot elbow sections
        for elbow_s in elbow_sections:
            k_val = elbow_s[0]
            intertia = elbow_s[1]
            plt.plot(k_val,
                     intertia,
                     'r*',)

        del inertias
        del k_models
        del elbow_cluster

        self.save_plot(f"Models/{model_name}",
                       f"All possible {model_name} Elbow's",)

        if display_visuals and self.__notebook_mode:
            plt.show()

        plt.close("all")

        # Get the counts of the center elbows
        center_elbow_count = pd.DataFrame({"Main Elbows": list(center_elbow_count.keys()),
                                           "Counts": list(center_elbow_count.values())})
        center_elbow_count.sort_values(by=['Counts'],
                                       ascending=False,
                                       inplace=True)

        self.save_table_as_plot(
            center_elbow_count,
            sub_dir=f"Models/{model_name}",
            filename="Center Elbow Count")

        # Get the counts of the proximity elbows
        proximity_elbow_count = pd.DataFrame({"Proximity Elbow": list(proximity_elbow_count.keys()),
                                              "Counts": list(proximity_elbow_count.values())})
        proximity_elbow_count.sort_values(by=['Counts'],
                                          ascending=False,
                                          inplace=True)

        self.save_table_as_plot(
            proximity_elbow_count,
            sub_dir=f"Models/{model_name}",
            filename="Proximity Elbow Count")

        # Set up graphic
        plt.figure(figsize=(13, 6))
        plt.title(f"Best of all {model_name} Elbows", fontsize=15)
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)



        # Find the most agreed upon elbow section
        average_elbow_inertias = elbow_inertias_matrix.mean(0)

        elbow_vote = []
        for vector in elbow_inertias_matrix:
            elbow_vote.append(
                np.absolute(vector - average_elbow_inertias).sum())

        best_elbow_index = np.array(elbow_vote).argmin()

        print(inertias_matrix[best_elbow_index].tolist()[0])

        # Plot most agreed upon elbow section
        plt.plot(ks,
                 inertias_matrix[best_elbow_index].tolist()[0],
                 '-o',
                 color='#367588')

        # Save the best cluster models
        best_clusters = []
        for model in elbow_models[best_elbow_index]:
            k_val = len(model.get_clusters())

            model_path = create_dir_structure(self.folder_path,
                                              f"Models/{model_name}/Clusters={k_val}")

            # Save model and meta data
            try:
                file_path = pickle_object_to_file(model,
                                                  model_path,
                                                  f"{model_name}_Clusters={str(k_val)}")

                pickle_object_to_file(self.__get_centers(model),
                                      model_path,
                                      f"Center points")


                dict_to_json_file({i:i for i in range(0,len(model.get_clusters()))},
                                  model_path,
                                  "Cluster Labels")

                self.__cluster_models_paths[
                    f"{model_name}_Clusters={str(k_val)}"] = file_path

            except:
                print(f"Something went wrong when trying to save the model: {model_name}")

            plt.plot(ks[k_val - 1],
                     inertias_matrix[best_elbow_index].tolist()[0][k_val - 1],
                     'r*')
            best_clusters.append(k_val)

        self.save_plot(f"Models/{model_name}",
                       f"Best of all {model_name} Elbows")

        if display_visuals and self.__notebook_mode:
            plt.show()
        plt.close("all")

        best_clusters.sort()

        if display_visuals and self.__notebook_mode:
            display(proximity_elbow_count)
            display(center_elbow_count)

        return best_clusters

    def __save_update_best_model_clusters(self):
        """
        Desc:
            Generates a directory structure for saving the suggested cluster
            amounts.
        """

        create_dir_structure(self.folder_path,
                             "_Extras")

        pickle_object_to_file(self.__models_suggested_clusters,
                              self.folder_path + "_Extras",
                              "All suggested clusters")
        write_object_text_to_file(self.__models_suggested_clusters,
                                  self.folder_path + "_Extras",
                                  "All suggested clusters")

        all_clusters = []
        for model_name, best_clusters in self.__models_suggested_clusters.items():
            write_object_text_to_file(best_clusters,
                                      self.folder_path + "_Extras",
                                      f"{model_name} suggested clusters")

            all_clusters += best_clusters


        write_object_text_to_file(round(sum(all_clusters) / len(all_clusters)),
                                  self.folder_path + "_Extras",
                                  "Average of suggested clusters")


    def __get_unique_random_indexes(self,
                                    k_val):
        """
        Desc:
            Helper function for pyclustering models that require list indexes
            to the scaled data.

        Args:
            k_val: int
                How many random center points or clusters are required.

        Returns:
            Returns a list of randomly indexes.
        """

        if len(self.__scaled) < k_val:
            raise ValueError("Can't generate more random indexes/center points "
                             "than avaliable.")

        random_indexes = set()
        while len(random_indexes) != k_val:

            index = random.randint(0, len(self.__scaled) - 1)

            if index not in random_indexes:
                random_indexes.add(index)

        return random_indexes


    def __determine_curve(self,
                          intertia):
        """
        Desc:
            When for finding the elbow of the graph it's important to understand
            if the curve of the intertia vector is convex or concaved

        Args:
            intertia: list of floats
                Cluster models evaluated by calculating their intertia.

        Returns:
            Returns a string that says either "concave" or "convex". Returns
            nothing if the intertia vector has a length less than 2.
        """

        # Can't determine concaved or convexed; return None
        if len(intertia) < 2:
            return None

        # Look at the vector in different lengths to determine the best shape
        convex_count = 0
        concave_count = 0
        for i in range(2, len(intertia) + 1):
            x = range(len(intertia))

            warnings.filterwarnings("ignore")

            # Calculate shape
            poly = np.polyfit(x[0:i],
                              intertia[0:i],
                              2)
            warnings.filterwarnings("default")

            # Derivative greater than 0 must be concaved
            if poly[0] >= 0:
                convex_count += 1
            else:
                concave_count += 1

        # Which shape was found more frequently within the vector.
        if concave_count >= convex_count:
            return "concave"
        else:
            return "convex"


    def __display_cluster_label_rank_graph(self,
                                           cluster_labels,
                                           counts,
                                           sub_dir="",
                                           display_visuals=True):
        """
        Desc:
            Darker colors have higher rankings (values)

        Args:
            cluster_labels: list of strings
                List of label names.

            counts: list of ints
                Counts for how many each cluster label had. Order must match
                'cluster_labels'.

            sub_dir: string
                Directory to create and write data to.

            display_visuals: bool
                If set to True will display graphic
        """
        plt.figure(figsize=(12, 7))

        palette = "PuBu"

        # Color ranking
        rank_list = np.argsort(-np.array(counts)).argsort()
        pal = sns.color_palette(palette, len(counts))
        palette = np.array(pal[::-1])[rank_list]

        plt.clf()

        plt.title("Cluster labels and data points count")

        ax = sns.barplot(x=cluster_labels,
                         y=counts,
                         palette=palette,
                         order=cluster_labels)

        # Labels for numerical count of each bar
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1}'.format(height),
                    ha="center")

        self.save_plot(sub_dir,
                       "Cluster labels and data points count")

        if display_visuals and self.__notebook_mode:
            plt.show()

        plt.close("all")

    def __get_color_name(self,
                         color):
        """
        Desc:
            Simple helper function for a simple switch statement.

        Args:
            color: char
                Single character value representing a real color name.

        Returns:
            A full color name.
        """

        if color == "b":
            return "Blue"

        elif color == "g":
            return "Green"

        elif color == "r":
            return "Red"

        elif color == "c":
            return "Cyan"

        elif color == "m":
            return "Magenta"

        elif color == "y":
            return "Yellow"

        elif color == "k":
            return "Black"

        elif color == "w":
            return "White"

        else:
            return color

    def __nearest_cluster(self,
                          center_points,
                          dp):
        """
        Desc:
            Simple helper function to get the right cluster label. Cluster label
            is determined by how close the given data point is to the center

        Args:
            center_points: list of lists of floats (list of data points)
                Cluster's center point.

            dp: list of floats, (data point)
                Data point
        """

        return np.argmin([euclidean_distance(dp, c)
                          for c in center_points])



