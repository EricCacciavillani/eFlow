from eflow._hidden.parent_objects import AutoModeler
from eflow.utils.sys_utils import pickle_object_to_file, create_dir_structure, write_object_text_to_file, check_if_directory_exists
from eflow.utils.eflow_utils import move_folder_to_eflow_garbage
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments

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
from kneed import DataGenerator, KneeLocator
import pandas as pd
import six
import random
import numpy as np
import copy
import os
from tqdm import tqdm


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
                 df_features,
                 project_sub_dir="",
                 project_name="Auto Clustering",
                 overwrite_full_path=None,
                 notebook_mode=False,
                 pca_perc=None):
        """
        Args:
            df: pd.Dataframe
                pd.Dataframe

            df_features: Dataframes type holder
                Dataframes type holder

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
        self.__all_cluster_models = dict()

        self.__df_features = copy.deepcopy(df_features)

        self.__notebook_mode = copy.deepcopy(notebook_mode)

        self.__models_suggested_clusters = dict()

        # --- Apply pca ---
        if pca_perc:

            # Create scaler object
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df)

            print("\nInspecting scaled results!")
            self.__inspect_feature_matrix(sub_dir="PCA",
                                          filename="Applied scaler results",
                                          matrix=scaled,
                                          feature_names=df.columns)

            pca, scaled = self.__visualize_pca_variance(scaled)

            # Generate "dummy" feature names
            pca_feature_names = ["PCA_Feature_" +
                                 str(i) for i in range(1,
                                                       len(df.columns) + 1)]

            print("\nInspecting applied scaler and pca results!")
            self.__inspect_feature_matrix(sub_dir="PCA",
                                          filename="Applied scaler and PCA results",
                                          matrix=scaled,
                                          feature_names=pca_feature_names)

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
            self.__inspect_feature_matrix(sub_dir="PCA",
                                          filename="Applied final sclaer to process.",
                                          matrix=scaled,
                                          feature_names=pca_feature_names)

            self.__scaled = scaled

        # Assumed PCA has already been applied; pass as matrix
        else:
            self.__scaled = df.values


    # --- Getters/Setters
    def get_scaled_data(self):
        """
        Desc:
            Gets the stored data

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
        return copy.deepcopy(self.__all_cluster_models)

    def delete_stored_data(self):
        """
        Desc:
            Removes the matrix data in order to save RAM when running
            analysis on the actual data.
        """
        del self.__scaled
        self.__scaled = None

    def visualize_hierarchical_clustering(self,
                                          linkage_methods=None,
                                          display_print=True,
                                          display_visuals=True):
        """
        Desc:
            Displays hierarchical cluster graphs with provided methods.

        Args:
            linkage_methods:
                All methods applied to the linkage

            display_print:
                Display print outputs

            display_visuals:
                Display plot data
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
                                         "Fuzzy C-means"],
                            repeat_operation=3,
                            max_k_value=15,
                            display_visuals=True):

        model_names = set(model_names)

        names_model_dict = {"K-Means":kmeans,
                            "K-Medians":kmedians,
                            "K-Medoids":kmedoids,
                            "Somsc":somsc,
                            "Cure":cure,
                            "Fuzzy C-means": fcm}

        # Iterate through passed model names
        for name in model_names:

            if name in names_model_dict.keys():

                # Only requires 1 elbow sequence
                if name == "Somsc" or name == "Cure":
                    best_clusters = self.__create_elbow_seq(name,
                                                            names_model_dict[name],
                                                            repeat_operation=1,
                                                            max_k_value=max_k_value,
                                                            display_visuals=display_visuals)
                else:
                    best_clusters = self.__create_elbow_seq(name,
                                                            names_model_dict[name],
                                                            repeat_operation=repeat_operation,
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
            Create multiple agglomerative models based on a list of
            'n_clusters' values and defined linkage methods.
        """

        if isinstance(n_cluster_list, int):
            n_cluster_list = [n_cluster_list]

        if not linkage_methods:
            linkage_methods = ["ward", "complete", "average", "single"]

        knn_graph = kneighbors_graph(
            self.__scaled, len(
                self.__scaled) - 1, include_self=False)

        for n_clusters in n_cluster_list:
            for connectivity in (None, knn_graph):

                for _, linkage in enumerate(linkage_methods):
                    model = AgglomerativeClustering(linkage=linkage,
                                                    connectivity=connectivity,
                                                    n_clusters=n_clusters)
                    model.fit(self.__scaled)
                    self.__all_cluster_models[
                        "AgglomerativeClustering_{0}_"
                        "cluster{1}_Connectivity{2}".format(
                            linkage,
                            n_clusters, connectivity is not None)] = model

                    print(
                        "Successfully generate Agglomerative model with "
                        "linkage {0} on n_clusters={1}".format(
                            linkage, n_clusters))

        # self.__models_suggested_clusters["Agglomerative models"] =

    def evaluate_all_models(self,
                            df,
                            df_features,
                            le_map=None,
                            show_extra=True,
                            find_nearest_on_cols=False,
                            zscore_low=-2,
                            zscore_high=2):
        """
            Loop through all models and evaluate the given model with
            'evaluate_model'. Read 'evaluate_model' to learn more.
        """

        for model_name, model in self.__all_cluster_models.items():
            self.evaluate_model(model_name=model_name,
                                model=model,
                                df=df,
                                df_features=df_features,
                                le_map=le_map,
                                find_nearest_on_cols=find_nearest_on_cols,
                                show_extra=show_extra,
                                zscore_low=zscore_low,
                                zscore_high=zscore_high)

            self.__vertical_spacing(5)
            print("----" * 20)

    def evaluate_model(self,
                       model_name,
                       model,
                       df,
                       df_features,
                       output_folder=None,
                       le_map=None,
                       show_extra=True,
                       find_nearest_on_cols=False,
                       zscore_low=-2,
                       zscore_high=2):
        """
        model_name:
            The string key to give the dict

        model:
            Cluster model type; it must have '.labels_' as an attribute

        df:
            Dataframe object

        df_features:
            DataFrameTypes object; organizes feature types into groups.

        output_folder:
            Sub directory to put the pngs

        le_map:
            Dict of dataframe cols to LabelEncoders

        show_extra:
            Show extra information from all functions

        find_nearest_on_cols:
                Allows columns to be converted to actual values found within
                the dataset.
                Ex: Can't have .7 in a bool column convert's it to 1.

                False: Just apply to obj columns and bool columns

                True: Apply to all columns

        zscore_low/zscore_high:
            Defines how the threshold to remove data points when profiling the
            cluster.


        The main purpose of 'evaluate_model' is to display/save tables/plots
        accoiated with describing the model's 'findings'.
        """

        df = copy.deepcopy(df)

        # Create folder structure for png outputs
        if not output_folder:
            output_path = str(model).split("(", 1)[0] + "/" + model_name
        else:
            output_path = output_folder + "/" + model_name

        # ---
        # self.__visualize_clusters(model=model,
        #                           output_path=output_path,
        #                           model_name=model_name)

        # ---
        df["Cluster_Name"] = model.labels_
        numerical_features = df_features.numerical_features()
        clustered_dataframes, shrunken_labeled_df = \
            self.__create_cluster_sub_dfs(
                df=df, model=model, numerical_features=numerical_features,
                zscore_low=zscore_low, zscore_high=zscore_high)

        rows_count, cluster_profiles_df = self.__create_cluster_profiles(
            clustered_dataframes=clustered_dataframes,
            shrunken_df=shrunken_labeled_df,
            numerical_features=df_features.numerical_features(),
            le_map=le_map,
            output_path=output_path,
            show=show_extra,
            find_nearest_on_cols=find_nearest_on_cols)

        # Check to see how many data points were removed during the profiling
        # stage
        print("Orginal points in dataframe: ", df.shape[0])
        print("Total points in all modified clusters: ", rows_count)
        print("Shrank by: ", df.shape[0] - rows_count)

        # In case to many data points were removed
        if cluster_profiles_df.shape[0] == 0:
            print(
                "The applied Z-scores caused the cluster profiles "
                "to shrink to far for the model {0}!".format(
                    model_name))

        # Display and save dataframe table
        else:
            display(cluster_profiles_df)
            self.__render_mpl_table(cluster_profiles_df, sub_dir=output_path,
                                    filename="All_Clusters",
                                    header_columns=0, col_width=2.0)

    def __create_cluster_profiles(self,
                                  clustered_dataframes,
                                  shrunken_df,
                                  numerical_features,
                                  le_map,
                                  output_path,
                                  find_nearest_on_cols=False,
                                  show=True):
        """
            Profile each clustered dataframe based off the given mean.
            Displays extra information in dataframe tables to be understand
            each cluster.

            find_nearest_on_cols:
                Allows columns to be converted to actual values found within
                the dataset.
                Ex: Can't have .7 in a bool column convert's it to 1.

                False: Just apply to obj columns and bool columns

                True: Apply to all columns
        """

        def find_nearest(numbers, target):
            """
                Find the closest fitting number to the target number
            """
            numbers = np.asarray(numbers)
            idx = (np.abs(numbers - target)).argmin()
            return numbers[idx]

        cluster_profiles_df = pd.DataFrame(columns=shrunken_df.columns).drop(
            'Cluster_Name', axis=1)
        rows_count = 0
        for cluster_identfier, cluster_dataframe in \
                clustered_dataframes.items():
            df = pd.DataFrame(columns=cluster_dataframe.columns)
            df = df.append(cluster_dataframe.mean(), ignore_index=True)
            df.index = [cluster_identfier]

            if cluster_dataframe.shape[0] <= 1:
                continue

            # Attempt to convert numbers found within the full set of data
            for col in cluster_dataframe.columns:
                if col not in numerical_features or find_nearest_on_cols:
                    df[col] = find_nearest(numbers=shrunken_df[
                        col].value_counts().index.tolist(),
                        target=df[col].values[0])

            # Evaluate cluster dataframe by dataframe
            eval_df = pd.DataFrame(columns=cluster_dataframe.columns)
            eval_df = eval_df.append(
                cluster_dataframe.mean(), ignore_index=True)
            eval_df = eval_df.append(
                cluster_dataframe.min(), ignore_index=True)
            eval_df = eval_df.append(
                cluster_dataframe.median(),
                ignore_index=True)
            eval_df = eval_df.append(
                cluster_dataframe.max(), ignore_index=True)
            eval_df = eval_df.append(
                cluster_dataframe.std(), ignore_index=True)
            eval_df = eval_df.append(
                cluster_dataframe.var(), ignore_index=True)
            eval_df.index = ["Mean", "Min", "Median",
                             "Max", "Standard Deviation", "Variance"]

            if show:
                print("Total found in {0} is {1}".format(
                    cluster_identfier, cluster_dataframe.shape[0]))
                self.__render_mpl_table(
                    df,
                    sub_dir=output_path,
                    filename=cluster_identfier +
                    "_Means_Rounded_To_Nearest_Real_Numbers",
                    header_columns=0,
                    col_width=4.0)

                self.__render_mpl_table(
                    eval_df,
                    sub_dir=output_path,
                    filename=cluster_identfier +
                    "_Eval_Df",
                    header_columns=0,
                    col_width=4.0)
                display(df)
                display(eval_df)
                self.__vertical_spacing(7)

            cluster_profiles_df = cluster_profiles_df.append(
                self.__decode_df(df, le_map))

            rows_count += cluster_dataframe.shape[0]

        return rows_count, cluster_profiles_df

    def __create_cluster_sub_dfs(self,
                                 df,
                                 model,
                                 numerical_features,
                                 zscore_low=-2,
                                 zscore_high=2):
        """
            Shrinks the clustered dataframe by rows based on outliers
            found within each cluster.

            Returns back a dict of dataframes with specficed clustered values
            alongside a full dataframe comprised of those clustered dataframes.
        """
        # Dataframe to analyze model 'better' choices
        shrunken_full_df = df.drop('Cluster_Name', axis=1).drop(df.index)

        # Store each sub-dataframe based on cluster label
        clustered_dataframes = dict()

        for cluster_label in set(model.labels_):
            cluster_df = df[df["Cluster_Name"] == cluster_label]
            # Ignore cluster with only one patient
            if len(cluster_df) <= 1:
                continue
            # ---
            zscore_cluster_df = cluster_df.drop(
                'Cluster_Name', axis=1).apply(zscore)

            # Check if cluster is only comprised of one data point
            if cluster_df.shape[0] > 1:

                # Iterate through all numerical features
                for numerical_feature in numerical_features:

                    nan_check = zscore_cluster_df[
                        numerical_feature].isnull().values.any()
                    # Check for nans
                    if not nan_check:
                        zscore_cluster_df = zscore_cluster_df[
                            zscore_cluster_df[numerical_feature] >= zscore_low]
                        zscore_cluster_df = zscore_cluster_df[
                            zscore_cluster_df[numerical_feature] <= zscore_high]

            # Dummy list of -1s alloc at given pos of 'zscore_cluster_df'
            # indexes
            reshaped_index = [-1] * len(df.index.values)

            for given_index in list(zscore_cluster_df.index.values):
                reshaped_index[given_index] = given_index

            # Pass back all vectors that passed the zscore test
            bool_array = pd.Series(reshaped_index).astype(int) == pd.Series(
                list(df.index.values)).astype(int)

            temp_cluster_df = df[bool_array].reset_index(drop=True)

            # Store in proper collection objs
            shrunken_full_df = shrunken_full_df.append(temp_cluster_df)

            clustered_dataframes[
                "Cluster:" + str(cluster_label)] = temp_cluster_df.drop(
                'Cluster_Name', axis=1)

        return clustered_dataframes, shrunken_full_df

    def __inspect_feature_matrix(self,
                                 sub_dir,
                                 filename,
                                 matrix,
                                 feature_names):
        """
            Creates a dataframe to quickly analyze a matrix
        """
        mean_matrix = np.mean(matrix, axis=0)
        std_matrix = np.std(matrix, axis=0)
        data_dict = dict()
        for index, feature_name in enumerate(feature_names):
            data_dict[feature_name] = [mean_matrix[index],
                                       std_matrix[index]]

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

    # Not created by me!
    # Created by my teacher: Narine Hall
    def __visualize_pca_variance(self, data):
        """
            Visualize PCA matrix feature importance
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

    def __display_rank_graph(self,
                             feature_names,
                             metric,
                             output_path,
                             model_name,
                             title="",
                             y_title="",
                             x_title="", ):
        """
            Darker colors have higher rankings (values)
        """
        plt.figure(figsize=(7, 7))

        # Init color ranking fo plot
        # Ref: http://tinyurl.com/ydgjtmty
        pal = sns.color_palette("GnBu_d", len(metric))
        rank = np.array(metric).argsort().argsort()
        ax = sns.barplot(y=feature_names, x=metric,
                         palette=np.array(pal[::-1])[rank])
        plt.xticks(rotation=0, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(x_title, fontsize=20, labelpad=20)
        plt.ylabel(y_title, fontsize=20, labelpad=20)
        plt.title(title, fontsize=15)
        plt.show()

    def __get_color_name(self,
                         color):

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

    def __nearest(self, clusters, x):
        return np.argmin([self.__distance(x, c) for c in clusters])

    def __distance(self, a, b):
        return np.sqrt(((a - b) ** 2).sum())

    def __get_unique_random_indexes(self,
                                    k_val):
        random_indexes = set()
        while len(random_indexes) != k_val:

            index = random.randint(0, len(self.__scaled) - 1)

            if index not in random_indexes:
                random_indexes.add(index)

        return random_indexes


    def __create_elbow_seq(self,
                           model_name,
                           model_instance,
                           repeat_operation,
                           max_k_value,
                           display_visuals):
        """
            Generate models based on the found 'elbow' of the interia values.
        """

        max_k_value += 1

        k_models = []
        inertias = []

        for elbow_seq_count in range(0,repeat_operation):
            tmp_inertias = []
            tmp_k_models = []

            if display_visuals:
                pbar = tqdm(range(1,max_k_value), desc=f"{model_name} Elbow Seq Count {elbow_seq_count + 1}")
            else:
                pbar = range(1,max_k_value)

            for k_val in pbar:

                if display_visuals:
                    pbar.set_postfix(model_count=k_val, refresh=True)


                model = self.__create_pyclustering_model(model_name=model_name,
                                                         model_instance=model_instance,
                                                         k_val=k_val)

                # Run cluster analysis and obtain results.
                model.process()
                final_centers = np.array(self.__get_centers(model))
                labels = [self.__nearest(final_centers, x) for x in self.__scaled]

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
                                    k_val):
        if model_name == "K-Medoids":
            model = model_instance(self.__scaled,
                                       [i for i in
                                        self.__get_unique_random_indexes(
                                            k_val)])
        elif model_name == "Somsc" or model_name == "Cure":
            model = model_instance(self.__scaled,
                               k_val)

        elif model_name == "K-Means" or model_name == "Fuzzy C-means":
            initial_centers = kmeans_plusplus_initializer(self.__scaled, k_val).initialize()
            model = model_instance(self.__scaled, initial_centers)

        else:
            # Create instance of K-Means algorithm with prepared centers.
            initial_centers = random_center_initializer(self.__scaled,
                                                        k_val).initialize()
            model = model_instance(self.__scaled, initial_centers)

        return model




    def __find_best_elbow_models(self,
                                 model_name,
                                 k_models,
                                 inertias,
                                 display_visuals=True):

        ks = range(1, len(inertias[0]) + 1)

        plt.figure(figsize=(13, 6))
        plt.title(f"All possible {model_name} Elbow's", fontsize=15)
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)

        elbow_inertias_matrix = None
        inertias_matrix = None
        elbow_models = []
        elbow_sections = []
        center_elbow_count = dict()
        proximity_elbow_count = dict()

        # Plot ks vs inertias
        for i in range(0,len(inertias)):

            elbow_cluster = KneeLocator(ks,
                                        inertias[i],
                                        curve='convex',
                                        direction='decreasing').knee

            if elbow_cluster == 1 or not elbow_cluster:
                print("Elbow was either one or None for the elbow seq.")
                continue

            plt.plot(ks,
                     inertias[i],
                     '-o',
                     color='#367588',
                     alpha=0.5)

            if str(elbow_cluster) not in center_elbow_count.keys():
                center_elbow_count[str(elbow_cluster)] = 1
            else:
                center_elbow_count[str(elbow_cluster)] += 1

            for k_val in [elbow_cluster - 1, elbow_cluster, elbow_cluster + 1]:
                elbow_sections.append([ks[k_val - 1],inertias[i][k_val - 1]])

                if str(k_val) not in proximity_elbow_count.keys():
                    proximity_elbow_count[str(k_val)] = 1
                else:
                    proximity_elbow_count[str(k_val)] += 1


            if isinstance(elbow_inertias_matrix, type(None)):
                inertias_matrix = np.matrix(inertias[i])
                elbow_inertias_matrix = np.matrix(inertias[i][elbow_cluster - 2:elbow_cluster + 1])

            else:
                inertias_matrix = np.vstack([inertias_matrix, inertias[i]])

                elbow_inertias_matrix = np.vstack(
                    [elbow_inertias_matrix, inertias[i][elbow_cluster - 2:elbow_cluster + 1]])

            elbow_models.append(k_models[i][elbow_cluster - 2:elbow_cluster + 1])

        for elbow in elbow_sections:
            k_val = elbow[0]
            intertia = elbow[1]
            plt.plot(k_val,
                     intertia,
                     'r*',)

        del inertias
        del k_models
        del elbow_cluster

        self.save_plot(f"Models/{model_name}",f"All possible {model_name} Elbow's",)

        if display_visuals and self.__notebook_mode:
            plt.show()
        plt.close("all")

        center_elbow_count = pd.DataFrame({"Main Knees": list(center_elbow_count.keys()),
                                           "Counts": list(center_elbow_count.values())})
        center_elbow_count.sort_values(by=['Counts'],
                                       ascending=False,
                                       inplace=True)

        self.save_table_as_plot(
            center_elbow_count,
            sub_dir=f"Models/{model_name}",
            filename="Center Elbow Count")

        proximity_elbow_count = pd.DataFrame({"Proximity Knees": list(proximity_elbow_count.keys()),
                                              "Counts": list(proximity_elbow_count.values())})
        proximity_elbow_count.sort_values(by=['Counts'],
                                          ascending=False,
                                          inplace=True)

        self.save_table_as_plot(
            proximity_elbow_count,
            sub_dir=f"Models/{model_name}",
            filename="Proximity Elbow Count")

        plt.figure(figsize=(13, 6))
        plt.title(f"Best of all {model_name} Elbows", fontsize=15)
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)

        average_elbow_inertias = elbow_inertias_matrix.mean(0)

        knee_vote = []
        for vector in elbow_inertias_matrix:
            knee_vote.append(
                np.absolute(vector - average_elbow_inertias).sum())

        best_elbow_index = np.array(knee_vote).argmin()

        plt.plot(ks,
                 inertias_matrix[best_elbow_index].tolist()[0],
                 '-o',
                 color='#367588')

        best_clusters = []
        for model in elbow_models[best_elbow_index]:
            k_val = len(model.get_clusters())

            self.__all_cluster_models[f"{model_name}_Cluster_" + str(k_val)] = model

            create_dir_structure(self.folder_path,
                                 f"Models/{model_name}/Clusters={k_val}")

            try:
                pickle_object_to_file(model,
                                      self.folder_path + f"Models/{model_name}/Clusters={k_val}",
                                      f"{model_name}_Cluster_" + str(k_val))
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





