# Expermental code designed to prove therories. NOT TO BE USED IN PRODUCTION YET!!!!

# Getting Sklearn Models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visuals Libs
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from IPython.display import display, HTML
import imageio
from IPython.display import Image

# Pandas lib
import pandas as pd

# Data Science Libs
from scipy.stats import zscore

# Math Libs
from scipy.spatial import distance
import numpy as np

# System Libs
import os, sys
import shutil
import math
import six
import time
import datetime
from functools import partial
import threading

# Misc Libs
import shelve
import copy
from collections import OrderedDict
from multiprocessing import Pool as ThreadPool
import multiprocessing as mp
from tqdm import tqdm

from eflow.Utils.Multi_Threading_Functions import *


class TargetSampleRemoval:
    def __init__(self,
                 df,
                 sample_target_dict,
                 columns_to_drop,
                 apply_pca=True,
                 pca_perc=.8,
                 project_name="Default",
                 overwrite_figure_path=None,
                 show_visuals=True,
                 ):
        """
        df:
            Must be a pandas dataframe object

        sample_target_dict:
            Column name(s) to value(s) in the dataframe to create a pandas
            dataframe with just those value(s).

        columns_to_drop:
            Column names to drop from the dataframe

        apply_pca:
            Data had already

        pca_perc:
            PCA cutoff point

        project_name:
            Starting folder name where the system

        overwrite_figure_path:
            Overwrites the absolute path for the images to be generated
        """

        def enum(**enums):
            return type('Enum', (), enums)

        if overwrite_figure_path:
            output_fig_sub_dir = overwrite_figure_path
        else:
            if pca_perc > 1:
                pca_perc = 1
            output_fig_sub_dir = "/Figures/" + project_name + \
                                 "/SampleRemoval_PCA_Features={0}".format(
                                     pca_perc)

        # Project directory structure
        self.__PROJECT = enum(
            PATH_TO_OUTPUT_FOLDER=''.join(
                os.getcwd().partition('/eflow')[0:1]) + output_fig_sub_dir)

        # Copy dataframes for later use
        df = copy.deepcopy(df)

        # Create dataframe of only target values
        for col, df_value in sample_target_dict.items():

            if isinstance(df_value, int):
                df_value = [df_value]

            for val in df_value:
                df = df[df[col] == val]

        for col in columns_to_drop:
            df.drop(columns=[col],
                    inplace=True)

        # --- Apply pca ---
        if apply_pca:

            # Create scaler object
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df)

            print("\nInspecting scaled results!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=df.columns)

            pca, scaled = self.__visualize_pca_variance(scaled, show_visuals)

            # Generate "dummy" feature names
            pca_feature_names = ["PCA_Feature_" +
                                 str(i) for i in range(1,
                                                       scaled.shape[1] + 1)]

            print("\nInspecting applied pca results!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=pca_feature_names)

            # Use only some of the features based on the PCA percentage
            if pca_perc < 1.0:
                cutoff_index = np.where(
                    pca.explained_variance_ratio_.cumsum() > pca_perc)[0][0]
            # Use all features
            else:
                cutoff_index = scaled.shape[1] - 1

            print(
                "After applying pca with a cutoff percentage of {0}%"
                " for the cumulative index. Using features 1 to {1}".format(
                    pca_perc, cutoff_index + 1))

            print("Old shape {0}".format(scaled.shape))

            scaled = scaled[:, :cutoff_index + 1]
            pca_feature_names = pca_feature_names[0: cutoff_index + 1]

            print("New shape {0}".format(scaled.shape))

            scaled = scaler.fit_transform(scaled)

            print("\nInspecting re-applied scaled results!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=pca_feature_names)

            self.__org_scaled = copy.deepcopy(scaled)
            self.__scaled = copy.deepcopy(scaled)

            self.__feature_weights = np.array(pca.explained_variance_ratio_[
                                              :scaled.shape[1]])

            print(self.__feature_weights)


            self.__feature_degress = (self.__feature_weights/self.__feature_weights.sum()) * 9

            print(self.__feature_degress)


        # Assumed PCA has already been applied; pass as matrix
        else:
            self.__scaled = df.values

        new_folder_path = ''.join(
            os.getcwd().partition('/eflow')[0:1]) + "/Figures/" + \
                          project_name + "/SampleRemoval_PCA_Features={0}".format(
            scaled.shape[1])

        if not os.path.exists(new_folder_path):
            os.rename(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                      new_folder_path)
        else:
            shutil.rmtree(self.__PROJECT.PATH_TO_OUTPUT_FOLDER)
        self.__PROJECT = enum(
            PATH_TO_OUTPUT_FOLDER=new_folder_path)
        self.__targeted_df = copy.deepcopy(df)

        # Init dummy variables to only be used for multithreading
        self.__index_array = None
        self.__total_indexes = None
        self.__tmp_reduced_scaled = None
        self.__all_dp_dist_list = None
        self.__removed_dps_dict = dict()
        self.__org_df_index_dict = None
        self.__saved_pic_paths_dict = dict()
        self.__applied_methods = set()
        self.__pbar = None

    def __weighted_eudis(self,
                         v1,
                         v2):
        dist = [((a - b) ** 2) * w for a, b, w in zip(v1, v2,
                                                      self.__feature_weights)]
        dist = math.sqrt(sum(dist))
        return dist

    def __rotate_point(self,
                       origin,
                       point,
                       angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.

        # Author link: http://tinyurl.com/y4yz5hco
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    # Not created by me!
    # Created by my teacher: Narine Hall
    # MODIFIED
    def __visualize_pca_variance(self,
                                 data,
                                 show_visuals):
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
        plt.ylabel('variance ratio')
        plt.xlabel('PCA feature')
        plt.tight_layout()
        self.__create_plt_png("PCA", "PCA_Feature_Variance_Ratio")
        if show_visuals:
            plt.show()
        plt.close()

        # ----
        plt.bar(range(1, len(pca.explained_variance_ratio_) + 1),
                pca.explained_variance_ratio_.cumsum())
        plt.xticks()
        plt.ylabel('cumulative sum of variances')
        plt.xlabel('PCA feature')
        plt.tight_layout()
        self.__create_plt_png("PCA", "PCA_Cumulative_Sum_of_Variances")
        if show_visuals:
            plt.show()
        plt.close()

        return pca, data

    def __check_create_figure_dir(self,
                                  sub_dir):
        """
            Checks/Creates required directory structures inside
            the parent directory figures.
        """

        directory_path = self.__PROJECT.PATH_TO_OUTPUT_FOLDER

        for dir in sub_dir.split("/"):
            directory_path += "/" + dir
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

        return directory_path

    def __create_plt_png(self,
                         sub_dir,
                         filename):
        """
            Saves the plt based image in the correct directory.
        """
        # Ensure directory structure is init correctly
        abs_path = self.__check_create_figure_dir(sub_dir)

        # Ensure file ext is on the file.
        if filename[-4:] != ".png":
            filename += ".png"

        fig = plt.figure(1)
        fig.savefig(abs_path + "/" + filename, bbox_inches='tight')

    def __inspect_feature_matrix(self,
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

        display(tmp_df)

        return tmp_df

    def remove_noise(self,
                     new_sample_amount,
                     zscore_high=2.0,
                     annotate=False,
                     apply_changes=False,
                     display_all_graphs=False,
                     show_gif=False,
                     shelve_relative_path=None,
                     create_visuals=True):

        new_sample_amount = int(new_sample_amount)

        if new_sample_amount >= self.__scaled.shape[0]:
            print("THROW ERROR HERE: Sample removal must be less then")
            return
        elif new_sample_amount <= 0:
            print("THROW ERROR HERE: Val must be a positive number!")
            return
        else:

            # Display graph before augmentation; Create centroid
            centroid = np.mean(self.__scaled, axis=0)
            column_list = [i for i in range(0, self.__scaled.shape[1])]

            df_index_scaled_dict = dict()

            reduced_scaled = np.column_stack(
                (self.__scaled, self.__targeted_df.index.values.reshape(
                    (self.__scaled.shape[0], 1)).astype(self.__scaled.dtype)))

            # Index to shape
            if not self.__org_df_index_dict:
                self.__org_df_index_dict = dict()
                for i, df_index in enumerate(self.__targeted_df.index.values):
                    df_index = reduced_scaled[i][-1]
                    self.__org_df_index_dict[df_index] = i

            for i, _ in enumerate(reduced_scaled):
                df_index = reduced_scaled[i][-1]
                df_index_scaled_dict[df_index] = i

            if create_visuals:

                if zscore_high:
                    folder_dir_name = "Data_Point_Removal_Noise_Zscore={0}".format(
                        zscore_high)
                else:
                    folder_dir_name = "Data_Point_Removal_Noise_Zscore=NaN"

                if apply_changes:
                    self.__applied_methods.add(folder_dir_name)

                self.__visualize_data_points(centroid=centroid,
                                             scaled_data=self.__scaled,
                                             new_sample_amount=new_sample_amount,
                                             annotate=annotate,
                                             apply_changes=apply_changes,
                                             output_path=folder_dir_name,
                                             called_from=sys._getframe().f_code.co_name,
                                             title="Starting point",
                                             display_all_graphs=display_all_graphs)

            dp_distances = np.zeros(len(reduced_scaled))

            if "Remove Noise" not in self.__removed_dps_dict.keys():
                self.__removed_dps_dict["Remove Noise"] = list()

        # Keep looping until new sample amount has been reached or
        # the distances are properly.
        while reduced_scaled.shape[0] > new_sample_amount:

            for index, dp in enumerate(reduced_scaled):
                dp_distances[index] = self.__weighted_eudis(
                    centroid, dp[:column_list[-1] + 1])

            farthest_dp_index = np.argmax(dp_distances)
            zscores_dp_distances = zscore(np.concatenate((
                dp_distances, np.array([self.__weighted_eudis(
                    centroid,self.__org_scaled[self.__org_df_index_dict[
                        dp_index]])
                                        for dp_index in
                                        self.__removed_dps_dict["Remove Noise"]
                                        ])), axis=0))

            if zscores_dp_distances[farthest_dp_index] >= zscore_high:

                # Add original dataframe index to the dict;
                # remove actual row from the data
                df_index = int(reduced_scaled[farthest_dp_index][-1])
                self.__removed_dps_dict["Remove Noise"].append(df_index)

                if shelve_relative_path:
                    shelf = shelve.open(shelve_relative_path)
                    shelf[shelve_relative_path.split("/")[-1]] = list(
                        self.__removed_dps_dict["Remove Noise"])
                    shelf.close()

                reduced_scaled = np.delete(reduced_scaled,
                                           farthest_dp_index,
                                           0)
                # Update centroid
                centroid = np.mean(reduced_scaled[:, column_list],
                                   axis=0)
                if create_visuals:

                    meta_data = dict()
                    meta_data["zscore"] = zscores_dp_distances[
                        farthest_dp_index]
                    meta_data["distance"] = dp_distances[
                        farthest_dp_index]

                    self.__visualize_data_points(centroid=centroid,
                                                 scaled_data=reduced_scaled[
                                                             :,
                                                             column_list],
                                                 new_sample_amount=new_sample_amount,
                                                 annotate=annotate,
                                                 apply_changes=apply_changes,
                                                 output_path=folder_dir_name,
                                                 meta_data=meta_data,
                                                 called_from=sys._getframe().f_code.co_name,
                                                 title="Data Removal: Noise reduction",
                                                 display_all_graphs=display_all_graphs)
                else:
                    print(
                        "Scaled size is now {0} and Z-Score {1:.2f}.".format(
                            reduced_scaled.shape[0],
                            zscores_dp_distances[farthest_dp_index]))
            # Break loop distances are below z-score val
            else:
                break

        if create_visuals:
            self.__create_gif_with_dp_amount(n_start=self.__scaled.shape[0],
                                             n_end=reduced_scaled.shape[0],
                                             folder_dir_name=folder_dir_name,
                                             filename="Noise Reduction",
                                             show_gif=show_gif)

        df_removal_indexes = copy.deepcopy(
            self.__removed_dps_dict["Remove Noise"])
        if apply_changes:
            self.__scaled = reduced_scaled[:, column_list]

            for i in df_removal_indexes:
                try:
                    self.__targeted_df.drop(i, inplace=True)
                except KeyError:
                    pass
        else:
            self.__removed_dps_dict.pop("Remove Noise", None)

        return df_removal_indexes

    def remove_similar(self,
                       new_sample_amount,
                       weighted_dist_value=1.0,
                       annotate=False,
                       apply_changes=False,
                       display_all_graphs=False,
                       show_gif=False,
                       shelve_relative_path=None,
                       create_visuals=True):

        self.__index_array = None
        self.__total_indexes = None
        self.__tmp_reduced_scaled = None
        self.__all_dp_dist_list = None
        self.__pbar = None
        self.__all_dp_dist_dict = None

        new_sample_amount = int(new_sample_amount)

        if new_sample_amount >= self.__scaled.shape[0]:
            print("THROW ERROR HERE: Sample removal must be less then")
        elif new_sample_amount <= 0:
            print("THROW ERROR HERE: Val must be a positive number!")
        else:

            df_index_scaled_dict = dict()
            reduced_scaled = np.column_stack(
                (self.__scaled, self.__targeted_df.index.values.reshape(
                    (self.__scaled.shape[0], 1)).astype(self.__scaled.dtype)))

            # Index to shape
            if not self.__org_df_index_dict:
                self.__org_df_index_dict = dict()
                for i, df_index in enumerate(self.__targeted_df.index.values):
                    df_index = reduced_scaled[i][-1]
                    self.__org_df_index_dict[df_index] = i

            for i, _ in enumerate(reduced_scaled):
                df_index = reduced_scaled[i][-1]
                df_index_scaled_dict[df_index] = i

            # Display graph before augmentation; Create centroid
            centroid = np.mean(self.__scaled, axis=0)
            column_list = [i for i in range(0, self.__scaled.shape[1])]

            for i, _ in enumerate(reduced_scaled):
                df_index = reduced_scaled[i][-1]
                df_index_scaled_dict[df_index] = i

            if create_visuals:

                if weighted_dist_value:
                    folder_dir_name = "Data_Point_Removal_Similar_Weight={0}".format(
                        weighted_dist_value)
                else:
                    folder_dir_name = "Data_Point_Removal_Similar_Weight=NaN"

                if apply_changes:
                    self.__applied_methods.add(folder_dir_name)
                self.__visualize_data_points(centroid=centroid,
                                             scaled_data=self.__scaled,
                                             new_sample_amount=new_sample_amount,
                                             annotate=annotate,
                                             apply_changes=apply_changes,
                                             output_path=folder_dir_name,
                                             called_from=sys._getframe().f_code.co_name,
                                             title="Starting point",
                                             display_all_graphs=display_all_graphs)

            starting_shape = reduced_scaled.shape[0]

            if "Remove Similar" not in self.__removed_dps_dict.keys():
                self.__removed_dps_dict["Remove Similar"] = list()

            farthest_dp_distance = None
            dp_distances = np.zeros(len(reduced_scaled))

            while reduced_scaled.shape[0] > new_sample_amount:
                # Following unconventional programming for multi threading
                # speed and memory increase
                self.__index_array = [i for i in
                                      range(0, len(reduced_scaled))]
                self.__total_indexes = len(self.__index_array)
                self.__tmp_reduced_scaled = copy.deepcopy(
                    reduced_scaled[:, column_list])

                if not farthest_dp_distance:
                    for index, dp in enumerate(self.__tmp_reduced_scaled):
                        dp_distances[index] = self.__weighted_eudis(
                            centroid, dp[:column_list[-1] + 1])

                    farthest_dp_distance = np.amax(dp_distances)
                    farthest_dp_distance *= weighted_dist_value

                removal_index, keep_index, smallest_dist = self.__shortest_dist_relationship(
                    centroid)

                if farthest_dp_distance < smallest_dist:
                    print("Target distance reached!!!")
                    break

                df_index = int(reduced_scaled[removal_index][-1])
                self.__removed_dps_dict["Remove Similar"].append(df_index)

                if shelve_relative_path:
                    shelf = shelve.open(shelve_relative_path)
                    shelf[shelve_relative_path.split("/")[-1]] =\
                        self.__removed_dps_dict["Remove Similar"]
                    shelf.close()

                # Remove from temp scaled
                reduced_scaled = np.delete(reduced_scaled,
                                           removal_index,
                                           0)
                # Update centroid
                centroid = np.mean(reduced_scaled[:, column_list],
                                   axis=0)

                if create_visuals:

                    meta_data = dict()
                    meta_data["kept_point"] = self.__tmp_reduced_scaled[
                        keep_index]

                    self.__visualize_data_points(centroid=centroid,
                                                 scaled_data=reduced_scaled[
                                                             :,
                                                             column_list],
                                                 new_sample_amount=new_sample_amount,
                                                 annotate=annotate,
                                                 apply_changes=apply_changes,
                                                 output_path=folder_dir_name,
                                                 called_from=sys._getframe().f_code.co_name,
                                                 meta_data=meta_data,
                                                 title="Data Removal: Similarity removal",
                                                 display_all_graphs=display_all_graphs)
                else:
                    print("Scaled size is now {0}.".format(
                        reduced_scaled.shape[0]))

        # De-init multithreading artifacts
        self.__index_array = None
        self.__total_indexes = None
        self.__tmp_reduced_scaled = None
        self.__all_dp_dist_list = None

        if create_visuals:
            self.__create_gif_with_dp_amount(n_start=starting_shape - 1,
                                             n_end=reduced_scaled.shape[0],
                                             folder_dir_name=folder_dir_name,
                                             filename="Similar Reduction",
                                             show_gif=show_gif)

        df_removal_indexes = copy.deepcopy(self.__removed_dps_dict["Remove Similar"])
        if apply_changes:
            self.__scaled = reduced_scaled[:, column_list]
            for i in df_removal_indexes:
                try:
                    self.__targeted_df.drop(i, inplace=True)
                except KeyError:
                    pass
        else:
            self.__removed_dps_dict.pop("Remove Similar", None)

        return df_removal_indexes

    def __find_dp_dist_mean(self,
                            target_index,
                            index_array,
                            scaled_data):
        distances = np.zeros(len(index_array))
        for index, dp_index in enumerate(
                filter(lambda x: x != target_index, index_array)):
            distances[index] = self.__weighted_eudis(
                scaled_data[target_index],
                scaled_data[dp_index])

        return np.mean(distances)

    def __shortest_dist_with_target(self,
                                    target_dp_index):
        """
            Finds the shortest distance between all dps based on its postional.
        """

        distances = np.zeros(self.__total_indexes - (target_dp_index + 1))
        for index, dp_index in enumerate(self.__index_array[
                                         target_dp_index + 1:]):
            distances[index] = self.__weighted_eudis(self.__tmp_reduced_scaled[
                                                         target_dp_index],
                                                     self.__tmp_reduced_scaled[
                                                         dp_index])
        shortest_dp_index = np.argmin(distances)

        return {
            target_dp_index: (
            self.__index_array[target_dp_index + 1:][shortest_dp_index],
            distances[shortest_dp_index])}

    def __shortest_dist_relationship(self,
                                     centroid):

        """
            Finds the two datapoints that have the smallest distance.
        """
        if not self.__all_dp_dist_list:

            total = 0
            for i in range(0,
                           self.__tmp_reduced_scaled.shape[0]):
                total += i

            print("The total time required is:", str(
                datetime.timedelta(seconds=total * 1.3e-5)))

            self.__all_dp_dist_list = find_all_distances_in_matrix(
                matrix=self.__tmp_reduced_scaled,
                index_array=self.__index_array,
                total_indexes=self.__total_indexes,
                feature_weights=self.__feature_weights)

        # :::ADD WEIGHTED DISTANCE IDEA HERE FUTURE ERIC:::

        all_shortest = [
            [target_dp_index,
             np.argmin(distances) + target_dp_index + 1,
             np.amin(distances)]
            for target_dp_index, distances in
            enumerate(self.__all_dp_dist_list)
            if len(distances) > 0]

        smallest_dps_relationship = min(all_shortest, key=lambda x: x[2])

        dp_1_index = smallest_dps_relationship[0]
        dp_2_index = smallest_dps_relationship[1]
        smallest_distance = smallest_dps_relationship[2]

        dp_1_dist = self.__weighted_eudis(self.__tmp_reduced_scaled[
                                              dp_1_index],
                                          centroid)

        dp_2_dist = self.__weighted_eudis(self.__tmp_reduced_scaled[
                                              dp_2_index],
                                          centroid)

        # Decide of the two dps which to remove
        removal_index = None
        keep_index = None
        if dp_1_dist < dp_2_dist:
            removal_index = dp_2_index
            keep_index = dp_1_index
        else:
            removal_index = dp_1_index
            keep_index = dp_2_index

        # Return distances values to everyone above the removed index
        for sub_removal_index, dp_index_key in enumerate(
                range(removal_index - 1, -1, -1)):
            self.__all_dp_dist_list[dp_index_key] = np.delete(
                self.__all_dp_dist_list[dp_index_key],
                sub_removal_index, 0)

        self.__all_dp_dist_list.pop(removal_index)

        # Return back the indexes and distance
        return removal_index, keep_index, smallest_distance

    def __create_gif_with_dp_amount(self,
                                    n_start,
                                    n_end,
                                    folder_dir_name,
                                    filename,
                                    flash_final_results=False,
                                    show_gif=False):
        """
            Generates a gif based on pre-generated images of sample removal.
        """

        if folder_dir_name:
            images = [imageio.imread(self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                                     folder_dir_name
                                     +
                                     "/Sample_removal_Visualized_Cluster_n={"
                                     "0}.png".format(i)) for i in range(n_start,
                                                                        n_end - 1,
                                                                        -1)]
        else:
            images = [imageio.imread(self.__saved_pic_paths_dict[i])
                                     for i in range(n_start,
                                                    n_end - 1,
                                                    -1)]

        if flash_final_results:

            if folder_dir_name:
                images += [imageio.imread(
                    self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                    folder_dir_name
                    + "/Sample_removal_Visualized_Cluster_n={0}.png".format(
                        n_start)), imageio.imread(
                    self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                    folder_dir_name
                    + "/Sample_removal_Visualized_Cluster_n={0}_White_Outed.png".format(
                        n_end))] * 4

            else:
                images += [imageio.imread(
                    self.__saved_pic_paths_dict[n_start]),
                              imageio.imread(
                                  self.__saved_pic_paths_dict[n_end])] * 4

        if folder_dir_name:
            imageio.mimsave(self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                            folder_dir_name +
                            "/{0}.gif".format(filename),
                            images,
                            duration=.68)
        else:
            imageio.mimsave(self.__PROJECT.PATH_TO_OUTPUT_FOLDER +
                            "/{0}.gif".format(filename),
                            images,
                            duration=.68)

        if show_gif:

            if folder_dir_name:
                display(Image(filename=self.__PROJECT.PATH_TO_OUTPUT_FOLDER +
                                       "/" + folder_dir_name +
                                       "/{0}.gif".format(filename)))
            else:
                display(Image(filename=self.__PROJECT.PATH_TO_OUTPUT_FOLDER +
                                       "/{0}.gif".format(filename)))

    def create_gif_with_dp_amount(self,
                                  n_start,
                                  n_end,
                                  filename=None,
                                  flash_final_results=False,
                                  show_gif=False):

        if not filename:
            filename = ""
            for given_method in self.__applied_methods:
                filename += given_method + " "

        self.__create_gif_with_dp_amount(n_start,
                                         n_end,
                                         folder_dir_name=None,
                                         filename=filename,
                                         flash_final_results=flash_final_results,
                                         show_gif=show_gif)

    def __visualize_data_points(self,
                                centroid,
                                scaled_data,
                                new_sample_amount,
                                annotate,
                                output_path,
                                called_from,
                                title,
                                apply_changes,
                                meta_data=None,
                                white_out_mode=False,
                                display_all_graphs=False,
                                no_print_output=False):
        """
            Creates visualization of clustering model on given data.
                Ex: The parameters are fairly self explanatory.
        """
        pl.figure(figsize=(8, 7))
        pl.title(title + "_n={0}".format(len(scaled_data),
                                         fontsize=15))

        plt.gcf().text(.94, .94, "Target_n_samples={0}".format(
            new_sample_amount), fontsize=12)

        # legennd_string = ""
        # for count, given_method in enumerate(self.__applied_methods):
        #     legennd_string += given_method.split('_', -1)[-1] + " "
        #
        #     if count % 2 == 0:
        #         legennd_string += "\n"
        #
        # plt.gcf().text(.91, 0.9,
        #                legennd_string,
        #                fontsize=12)

        cell_information = np.array([])
        row_information = np.array([])

        for method_count, given_method in enumerate(self.__applied_methods):
            cell_information = np.append(cell_information,
                                         given_method)
            row_information = np.append(row_information,
                                        "Removal Process {0}".format(
                                            method_count))
        cell_information = cell_information.reshape(len(
            self.__applied_methods), 1)

        # plt.axis("off")

        # the_table = plt.table(cellText=cell_information,
        #                       rowLabels=row_information,
        #                       colLabels=np.array(["Table"]),
        #                       colWidths=[0.5] * 3,
        #                       loc='center left',
        #                       bbox=[1.3, -0.5, 0.5, 0.5],
        #                       fontsize=14)
        plt.subplots_adjust(bottom=0.3)
        # plt.show()

        # Plot existing data points
        for i in range(0, scaled_data.shape[0]):

            rotation_degrees = (
                    (abs(centroid - scaled_data[i])/2) * self.__feature_degress).sum()

            px, py = self.__rotate_point(np.array([0,0]),
                                         np.array(
                                             [0,
                                              self.__weighted_eudis(
                                                  scaled_data[i], centroid)]),
                                         rotation_degrees)

            pl.scatter(px,
                       py,
                       c="#0080FF",
                       marker='o',
                       label="Existing data point")

            # plt.annotate('{0}'.format(i), xy=(np.mean(scaled_data[i]),
            #            distance.euclidean(scaled_data[i], centroid)),
            #              xytext=(np.mean(scaled_data[i]),
            #                      distance.euclidean(scaled_data[i], centroid)))

        # Plot data points removed from noise removal
        for key_name,list_of_removed_indexes in self.__removed_dps_dict.items():

            last_index = len(list_of_removed_indexes) - 1
            for index, dp_index in enumerate(list_of_removed_indexes):
                if white_out_mode:

                    rotation_degrees = (
                            (abs(centroid - dp)/2) * self.__feature_degress).sum()

                    px, py = self.__rotate_point(np.array([0,0]),
                                                 np.array(
                                                     [0,
                                                      self.__weighted_eudis(
                                                          self.__org_scaled[
                                                              dp_index])]),
                                                 rotation_degrees)

                    pl.scatter(px,
                               py,
                               c="#ffffff",
                               marker='X',
                               alpha=0)
                else:

                    if key_name == "Remove Noise":

                        dp = self.__org_scaled[self.__org_df_index_dict[
                            dp_index]]

                        rotation_degrees = (
                                (abs(centroid - dp)/2) * self.__feature_degress).sum()

                        px, py = self.__rotate_point(np.array([0,0]),
                                                     np.array([0,
                                                               self.__weighted_eudis(
                                                                   dp,
                                                                   centroid)]),
                                                     rotation_degrees)

                        pl.scatter(px,
                                   py,
                                   c="#00A572",
                                   marker='X',
                                   label="Noise Removal")

                        if annotate and meta_data \
                                and index == last_index \
                                and called_from == "remove_noise":

                            dp = self.__org_scaled[
                                self.__org_df_index_dict[dp_index]]

                            # Find the correct angle to have the text and annotated line match
                            dp_centroid_dist = self.__weighted_eudis(dp,
                                                                     centroid)

                            dy = (0 - py)
                            dx = (0 - px)
                            rotn = np.degrees(np.arctan2(dy, dx))
                            trans_angle = plt.gca().transData.transform_angles(
                                np.array((rotn,)), np.array((dx,
                                                             dy)).reshape(
                                    (1, 2)))[0]
                            # Fix text representation on the given angle
                            if trans_angle > 90:
                                trans_angle = trans_angle - 180

                            if trans_angle < -90:
                                trans_angle = trans_angle + 180

                            # Spacing for formatting
                            spacing = "\n" * 2
                            if trans_angle < 0:
                                spacing = "\n" * 4

                            # Create line
                            plt.annotate(' ', xy=(px,
                                                  py),
                                         xytext=(0,
                                                 0),
                                         rotation=trans_angle,
                                         ha='center',
                                         va='center',
                                         rotation_mode='anchor',
                                         arrowprops={'arrowstyle': '<->',
                                                     'shrinkA': .4,
                                                     'shrinkB': 4.5}
                                         )

                            # Create text
                            plt.annotate(
                                spacing + 'zscore={0:.2f} , Dist:{1:.2f}\n'.format(
                                    meta_data["zscore"],
                                    meta_data["distance"]),
                                xy=(0, 0),
                                rotation_mode='anchor',
                                va='center',
                                ha='center',
                                rotation=trans_angle)

                    elif key_name == "Remove Similar":
                        dp = self.__org_scaled[self.__org_df_index_dict[
                            dp_index]]

                        rotation_degrees = (
                                (abs(centroid - dp)/2) * self.__feature_degress).sum()

                        px, py = self.__rotate_point(np.array([0, 0]),
                                                     np.array([0,
                                                               self.__weighted_eudis(
                                                                   dp,
                                                                   centroid)]),
                                                     rotation_degrees)

                        pl.scatter(px,
                                   py,
                                   c="#8A2BE2",
                                   marker='X',
                                   label="Similar Removal")

                        if annotate and meta_data \
                                and index == last_index \
                                and called_from == "remove_similar":
                            rotation_degrees = (
                                    (abs(centroid - meta_data["kept_point"])/2) *
                                    self.__feature_degress).sum()
                            meta_px, meta_py = self.__rotate_point(np.array([0, 0]),
                                                                   np.array([0,
                                                                             self.__weighted_eudis(
                                                                                 meta_data[
                                                                                     "kept_point"],
                                                                                 centroid)]),
                                                                   rotation_degrees)
                            # Create line
                            plt.annotate(' ',
                                         xy=(px,
                                             py),
                                         xytext=(
                                             meta_px,
                                             meta_py),
                                         ha='center',
                                         va='center',
                                         rotation_mode='anchor',
                                         arrowprops={'arrowstyle': '<->',
                                                     'shrinkA': .4,
                                                     'shrinkB': 4.5}
                                         )
                            from scipy.spatial import distance
                            print(distance.euclidean((px,py),(meta_px, meta_py)))

        # Plot centroid
        pl.scatter(0, 0,
                   c="r", marker="D",
                   label="Centroid")
        # Author: http://tinyurl.com/yxvd33t2
        # Removes all duplicated handles and labels of the labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   loc='center left', bbox_to_anchor=(1, 0.92))

        filename_format = None
        if white_out_mode:
            filename_format = "Sample_removal_Visualized_Cluster_n={0}_White_Outed".format(
                                      len(scaled_data))

        else:
            filename_format = "Sample_removal_Visualized_Cluster_n={0}".format(
                                      len(scaled_data))

        self.__create_plt_png(output_path,
                              filename_format)

        if apply_changes:
            self.__saved_pic_paths_dict[
                len(scaled_data)] = self.__PROJECT.PATH_TO_OUTPUT_FOLDER + \
                                    "/" + output_path + "/" + filename_format\
                                    + ".png"
        if display_all_graphs:
            plt.show()
        else:
            # if self.__scaled.shape[0] > scaled_data.shape[0] and \
            #         not no_print_output:
            #     if new_dp_meta_noise_removal:
            #         print("Scaled size is now {0}".format(
            #             scaled_data.shape[
            #                 0]) + " and Z-Score of {0:.2f}".format(
            #             new_dp_meta_noise_removal[1]))

            # print("Scaled size is now {0}".format(scaled_data.shape[0]
            #                                               ))
            pass

        plt.close()

    # --- Misc
    # # I am this lazy yes...
    # def __vertical_spacing(self, spaces=1):
    #     for _ in range(0, spaces):
    #         print()

    # --- Getters/Setters
    def get_scaled_data(self):
        return copy.deepcopy(self.__scaled)

    def testing_table_data(self):
        return copy.deepcopy(self.__applied_methods)



