
from eflow._hidden.parent_objects import AutoModeler

# Getting Sklearn Models
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import linkage, dendrogram,set_link_color_palette
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Visuals libs
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from IPython.display import display, HTML

# Misc
from collections import Counter
from scipy.stats import zscore
from kneed import DataGenerator, KneeLocator
import pandas as pd
import six
import numpy as np
import copy

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class AutoCluster(AutoModeler):

    def __init__(self,
                 df,
                 df_features,
                 project_sub_dir="",
                 project_name="Auto Clustering",
                 overwrite_full_path=None,
                 notebook_mode=False):

        AutoModeler.__init__(self,
                             f'{project_sub_dir}/{project_name}',
                             overwrite_full_path)

        # Define model
        self.__all_cluster_models = dict()

        self.__df_features = copy.deepcopy(df_features)

        self.__notebook_mode = copy.deepcopy(notebook_mode)

        pca_perc = .8

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
        return copy.deepcopy(self.__scaled)

    def get_all_cluster_models(self):
        return copy.deepcopy(self.__all_cluster_models)

    def visualize_hierarchical_clustering(self,
                                          linkage_methods=None,
                                          display_print=True,
                                          display_visuals=True):
        """
            Displays hierarchical cluster graphs with provided methods.
        """

        if not linkage_methods:
            linkage_methods = ["complete",
                               "single",
                               "weighted",
                               "ward",
                               "average",
                               "centroid",
                               "median"]

        for method in linkage_methods:

            if display_print:
                print(f"Creating graphic for Hierarchical Clustering Method: {method}...")

            plt.figure(figsize=(12, 7))
            # Calculate the linkage: mergings
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
             # "Orange":"#cc7722"}

            set_link_color_palette(None)

            # Plot the dendrogram, using varieties as labels
            color_list = dendrogram(mergings,
                                    labels=list(range(0, len(self.__scaled,))),
                                    leaf_rotation=90,
                                    leaf_font_size=3)["color_list"]

            method = method.capitalize()

            plt.title(f"Hierarchical Clustering Method : \'{method}\'")
            self.save_plot("Hierarchical Clustering",
                           f"Hierarchical Clustering Method {method} without legend")

            color_cluster_count = dict()
            last_color = None
            known_colors = set()
            color_cluster_order = list()
            seq_len = 0
            i = 0
            for color in copy.deepcopy(color_list):

                color = self.__get_color_name(color)
                if color in known_colors:
                    color_list[i] = f"{color} cluster {color_cluster_count[color]}"
                else:
                    color_list[i] = f"{color} cluster 0"

                if color_list[i] not in color_cluster_order:
                    color_cluster_order.append(color_list[i])

                if last_color:
                    # Sequence of color has yet to be broken
                    if last_color == color:

                        if seq_len <= 2:
                            seq_len += 1

                    # Sequence break
                    else:
                        if seq_len > 1:

                            if last_color not in known_colors:
                                known_colors.add(last_color)

                            if last_color not in color_cluster_count.keys():
                                color_cluster_count[last_color] = 1
                            else:
                                color_cluster_count[last_color] += 1
                        else:
                            color_list.pop(i-1)
                            i -= 1

                        seq_len = 0

                last_color = color
                i += 1

            counter_object = Counter(color_list)
            cluster_color_count = dict()
            import matplotlib.patches as mpatches
            handles = []
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


            plt.legend(handles=handles,
                       loc=(1.01, .72))

            if display_visuals:
                plt.show()

            plt.close('all')

            self.save_plot("Hierarchical Clustering",
                           f"Hierarchical Clustering Method {method} with legend")


    def __visualize_clusters(self, model, output_path, model_name=""):
        """
            Creates visualization of clustering model on given data.
        """
        markers = ["+", "*", "X", "o", "v", "P", "H", "4", "p", "D", "s",
                   "1", "x", "d", "_"]
        colors = ['b', 'g', 'r', 'c', 'm', 'y',
                  '#007BA7', '#ff69b4', '#CD5C5C', '#7eab19', '#1a4572',
                  '#2F4F4F', '#4B0082', '#d11141', '#5b2504']

        # Display ranking on color based on amount data points per cluster
        unique, counts = np.unique(model.labels_, return_counts=True)
        cluster_names = ["Cluster:" + str(cluster_label)
                         for cluster_label in unique]
        self.__display_rank_graph(feature_names=cluster_names,
                                  metric=counts,
                                  title=model_name,
                                  output_path=output_path,
                                  model_name=model_name,
                                  y_title="Clusters",
                                  x_title="Found per cluster")
        pl.figure(figsize=(8, 7))

        # Display clustered graph
        cluster_array = list(range(0, len(cluster_names)))
        scaled_cluster_label = np.hstack(
            (self.__scaled, np.reshape(
                model.labels_.astype(int), (self.__scaled.shape[0], 1))))
        for i in range(0, scaled_cluster_label.shape[0]):
            cluster_label = int(scaled_cluster_label[i][-1])
            cluster_array[cluster_label] = pl.scatter(
                scaled_cluster_label[i, 0], scaled_cluster_label[i, 1],
                c=colors[cluster_label], marker=str(markers[cluster_label]))

        pl.legend(cluster_array, cluster_names)
        pl.title(model_name + ' visualized with data', fontsize=15)
        self.__image_processing_utils(output_path,
                              model_name + "_Visualized_Cluster")
        plt.show()
        plt.close()
        pl.close()

    def create_kmeans_models(self,
                             n_cluster_list,
                             random_state=None):
        """
            Create kmeans models based on a list of 'n_clusters' values
        """

        if isinstance(n_cluster_list, int):
            n_cluster_list = [n_cluster_list]

        for k_val in n_cluster_list:
            self.__all_cluster_models["Kmeans_Cluster_" + str(k_val)] = KMeans(
                n_clusters=k_val, random_state=random_state).fit(self.__scaled)
            print(
                "Successfully generate Kmeans model on "
                "pre_defined_k={0}".format(k_val))

    def create_kmeans_models_with_elbow_graph(self,
                                              random_state=None):
        """
            Generate models based on the found 'elbow' of the interia values.
        """

        ks = range(1, 15)
        inertias = []

        for k in ks:
            # Create a KMeans instance with k clusters: model
            model = KMeans(n_clusters=k,
                           random_state=random_state).fit(self.__scaled)

            # Append the inertia to the list of inertias
            inertias.append(model.inertia_)

        # Plot ks vs inertias
        plt.figure(figsize=(13, 6))
        plt.plot(ks, inertias, '-o')
        plt.xlabel('Number of clusters, k')
        plt.ylabel('Inertia')
        plt.xticks(ks)
        plt.title("Kmeans Elbow", fontsize=15)

        a = KneeLocator(inertias, ks, curve='convex', direction='decreasing')
        elbow_index = np.where(inertias == a.knee)
        elbow_index = elbow_index[0][0] + 1

        for k_val in [elbow_index - 1, elbow_index, elbow_index + 1]:
            self.__all_cluster_models["Kmeans_Cluster_" + str(k_val)] = KMeans(
                n_clusters=k_val, random_state=random_state).fit(self.__scaled)
            plt.plot(ks[k_val - 1], inertias[k_val - 1], 'r*')

            print(
                "Successfully generate Kmeans model on k_val={0}".format(k_val)
            )

        self.__image_processing_utils("Kmeans",
                              "Kmeans_Visualized_Cluster")
        plt.show()
        plt.close()

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
        self.__visualize_clusters(model=model,
                                  output_path=output_path,
                                  model_name=model_name)

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

    # Takes a dict of LabelEncoders to decode the dataframe
    def __decode_df(self, df, le_map):

        df = copy.deepcopy(df)
        decode_cols = list(le_map.keys())
        df[decode_cols] = df[decode_cols].apply(
            lambda x: le_map[x.name].inverse_transform(x))

        return df

    # Not created by me!
    # Author: http://tinyurl.com/y2hjhbwf
    def __render_mpl_table(
            self,
            data,
            sub_dir,
            filename,
            col_width=3.0,
            row_height=0.625,
            font_size=14,
            header_color='#40466e',
            row_colors=['#f1f1f2', 'w'],
            edge_color='w',
            bbox=[0, 0, 1, 1],
            header_columns=0,
            ax=None,
            **kwargs):

        if ax is None:
            size = (np.array(
                data.shape[::-1]) + np.array([0, 1])) * np.array([col_width,
                                                                  row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(
            cellText=data.values,
            bbox=bbox,
            colLabels=data.columns,
            **kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)

        for k, cell in six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])

        self.__image_processing_utils(sub_dir, filename)

        plt.close()

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


