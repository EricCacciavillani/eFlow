# Getting Sklearn Models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visuals Libs
import matplotlib.pyplot as plt
import pylab as pl
import seaborn as sns
from IPython.display import display, HTML
import imageio

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


def find_all_dist_with_target(matrix,
                              index_array,
                              total_indexes,
                              target_dp_index):
    """
        Finds all distances between the target and the other points.
    """
    distances = np.zeros(total_indexes - (target_dp_index + 1))
    for index, dp_index in enumerate(index_array[
                                     target_dp_index + 1:]):
        distances[index] = distance.euclidean(matrix[
                                                  target_dp_index],
                                              matrix[dp_index])
        # self.__pbar.update(1)
    # shortest_dp_index = np.argmin(distances)

    all_distances_to_target = dict()

    all_distances_to_target[target_dp_index] = distances

    return all_distances_to_target


def find_all_distances_in_matrix(matrix,
                                 index_array,
                                 total_indexes):
    pool = ThreadPool(mp.cpu_count()-2)

    func = partial(find_all_dist_with_target, matrix, index_array,
                   total_indexes)
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


class SampleRemoval:
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

            self.__scaled = scaled

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

        self.__df_index_values = df.index.values

        # Init dummy variables to only be used for multithreading
        self.__index_array = None
        self.__total_indexes = None
        self.__tmp_reduced_scaled = None
        self.__all_dp_dist_list = None
        self.__pbar = None

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

        directory_pth = self.__PROJECT.PATH_TO_OUTPUT_FOLDER

        for dir in sub_dir.split("/"):
            directory_pth += "/" + dir
            if not os.path.exists(directory_pth):
                os.makedirs(directory_pth)

        return directory_pth

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

    def remove_samples(self,
                       new_sample_amount,
                       zscore_high=2,
                       weighted_dist_value=1.0,
                       annotate=False,
                       remove_noise=True,
                       remove_similar=True,
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
        elif remove_noise == False and remove_similar == False:
            print("THROW ERROR HERE: At least one operation must be made!")
        else:
            # Store data for removal
            removed_dps_dict = dict()

            # Stored removed datapoints for visualizations
            noise_removal_dps_dict = dict()
            similarity_dps_dict = dict()

            df_index_scaled_dict = dict()
            # Index to shape
            for i, df_index in enumerate(self.__df_index_values):
                df_index_scaled_dict[df_index] = i

            if not remove_noise:
                folder_dir_name = "Data_Point_Removal_Weight={1}".format(
                    zscore_high, weighted_dist_value)

            elif not remove_similar:
                folder_dir_name = "Data_Point_Removal_Zscore={0}".format(
                    zscore_high, weighted_dist_value)

            else:
                folder_dir_name = "Data_Point_Removal_Zscore={0}_Weight={1}".format(
                    zscore_high, weighted_dist_value)

            # Display graph before augmentation; Create centroid
            centroid = np.mean(self.__scaled, axis=0)
            column_list = [i for i in range(0, self.__scaled.shape[1])]

            reduced_scaled = np.column_stack(
                (self.__scaled, self.__df_index_values.reshape(
                    (self.__scaled.shape[0], 1)).astype(self.__scaled.dtype)))

            if create_visuals:
                self.__visualize_data_points(centroid=centroid,
                                             scaled_data=self.__scaled,
                                             noise_removal_dps=[],
                                             similar_removal_dps=[],
                                             new_sample_amount=new_sample_amount,
                                             zscore_high=zscore_high,
                                             weighted_dist_value=weighted_dist_value,
                                             annotate=annotate,
                                             output_path=folder_dir_name,
                                             title="Starting point",
                                             remove_noise=remove_noise,
                                             remove_similar=remove_similar,
                                             display_all_graphs=display_all_graphs)

            if remove_noise:

                dp_distances = np.zeros(len(reduced_scaled))

                # Keep looping until new sample amount has been reached or
                # the distances are properly.
                while reduced_scaled.shape[0] > new_sample_amount:

                    for index, dp in enumerate(reduced_scaled):
                        dp_distances[index] = distance.euclidean(
                            centroid, dp[:column_list[-1] + 1])

                    farthest_dp_index = np.argmax(dp_distances)
                    zscores_dp_distances = zscore(np.concatenate((
                        dp_distances, np.array([distance.euclidean(centroid,
                                                                   self.__scaled[
                                                                       dp_index])
                                                for dp_index in
                                                list(removed_dps_dict.values())
                                                ])), axis=0))

                    if zscores_dp_distances[farthest_dp_index] >= zscore_high:

                        farthest_dp = reduced_scaled[farthest_dp_index][
                                      :column_list[-1] + 1]

                        # Add original dataframe index to the dict;
                        # remove actual row from the data

                        df_index = int(reduced_scaled[farthest_dp_index][-1])
                        removed_dps_dict[df_index] = df_index_scaled_dict[
                            df_index]

                        if shelve_relative_path:
                            shelf = shelve.open(shelve_relative_path)
                            shelf[shelve_relative_path.split("/")[-1]] = list(
                                removed_dps_dict.keys())
                            shelf.close()

                        if create_visuals:
                            noise_removal_dps_dict[df_index] = \
                            df_index_scaled_dict[df_index]

                        reduced_scaled = np.delete(reduced_scaled,
                                                   farthest_dp_index,
                                                   0)
                        # Update centroid
                        centroid = np.mean(reduced_scaled[:, column_list],
                                           axis=0)
                        if create_visuals:
                            self.__visualize_data_points(centroid=centroid,
                                                         scaled_data=reduced_scaled[
                                                                     :,
                                                                     column_list],
                                                         noise_removal_dps=list(
                                                             noise_removal_dps_dict.values()),
                                                         similar_removal_dps=[],
                                                         new_sample_amount=new_sample_amount,
                                                         zscore_high=zscore_high,
                                                         weighted_dist_value=weighted_dist_value,
                                                         annotate=annotate,
                                                         output_path=folder_dir_name,
                                                         new_dp_meta_noise_removal=(
                                                         farthest_dp,
                                                         zscores_dp_distances[
                                                             farthest_dp_index],
                                                         dp_distances[
                                                             farthest_dp_index]),
                                                         title="Data Removal: Noise reduction",
                                                         remove_noise=remove_noise,
                                                         remove_similar=remove_similar,
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
                    self.__create_gif_dp_amount(n_start=self.__scaled.shape[0],
                                                n_end=reduced_scaled.shape[0],
                                                folder_dir_name=folder_dir_name,
                                                filename="Noise Reduction",
                                                show_gif=show_gif)

            if remove_similar:

                starting_shape = reduced_scaled.shape[0]
                
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
                            dp_distances[index] = distance.euclidean(
                                centroid, dp[:column_list[-1] + 1])

                        farthest_dp_distance = np.amax(dp_distances)
                        farthest_dp_distance *= weighted_dist_value

                    removal_index, keep_index, smallest_dist = self.__shortest_dist_relationship(
                        centroid)
                    
                    if farthest_dp_distance < smallest_dist:
                        print("Target distance reached!!!")
                        break

                    new_dp_meta_similar_removal = (
                    self.__tmp_reduced_scaled[removal_index],
                    self.__tmp_reduced_scaled[keep_index])

                    df_index = int(reduced_scaled[removal_index][-1])
                    removed_dps_dict[df_index] = df_index_scaled_dict[df_index]

                    if create_visuals:
                        similarity_dps_dict[df_index] = df_index_scaled_dict[
                            df_index]

                    if shelve_relative_path:
                        shelf = shelve.open(shelve_relative_path)
                        shelf[shelve_relative_path.split("/")[-1]] = list(
                            removed_dps_dict.keys())
                        shelf.close()

                    # Remove from temp scaled
                    reduced_scaled = np.delete(reduced_scaled,
                                               removal_index,
                                               0)
                    # Update centroid
                    centroid = np.mean(reduced_scaled[:, column_list],
                                       axis=0)

                    if create_visuals:
                        self.__visualize_data_points(centroid=centroid,
                                                     scaled_data=reduced_scaled[
                                                                 :,
                                                                 column_list],
                                                     noise_removal_dps=list(
                                                         noise_removal_dps_dict.values()),
                                                     similar_removal_dps=list(
                                                         similarity_dps_dict.values()),
                                                     new_sample_amount=new_sample_amount,
                                                     zscore_high=zscore_high,
                                                     weighted_dist_value=weighted_dist_value,
                                                     annotate=annotate,
                                                     output_path=folder_dir_name,
                                                     new_dp_meta_similar_removal=new_dp_meta_similar_removal,
                                                     title="Data Removal: Similarity removal",
                                                     remove_noise=remove_noise,
                                                     remove_similar=remove_similar,
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
                self.__create_gif_dp_amount(n_start=starting_shape - 1,
                                            n_end=reduced_scaled.shape[0],
                                            folder_dir_name=folder_dir_name,
                                            filename="Similar Reduction",
                                            show_gif=show_gif)

            if remove_similar and remove_noise and create_visuals:
                self.__visualize_data_points(centroid=centroid,
                                             scaled_data=reduced_scaled[:,
                                                         column_list],
                                             noise_removal_dps=list(
                                                 noise_removal_dps_dict.values()),
                                             similar_removal_dps=list(
                                                 similarity_dps_dict.values()),
                                             new_sample_amount=new_sample_amount,
                                             zscore_high=zscore_high,
                                             weighted_dist_value=weighted_dist_value,
                                             annotate=annotate,
                                             output_path=folder_dir_name,
                                             new_dp_meta_similar_removal=None,
                                             title="Final Result",
                                             remove_noise=remove_noise,
                                             remove_similar=remove_similar,
                                             white_out_mode=True,
                                             no_print_output=True,
                                             display_all_graphs=display_all_graphs)

                self.__create_gif_dp_amount(n_start=self.__scaled.shape[0],
                                            n_end=reduced_scaled.shape[0],
                                            folder_dir_name=folder_dir_name,
                                            filename="Noise and Similar Reduction",
                                            flash_final_results=True,
                                            show_gif=show_gif)

            if apply_changes:
                self.__scaled = reduced_scaled[:, column_list]

            return list(removed_dps_dict.keys())

    def __find_dp_dist_mean(self,
                            target_index,
                            index_array,
                            scaled_data):
        distances = np.zeros(len(index_array))
        for index, dp_index in enumerate(
                filter(lambda x: x != target_index, index_array)):
            distances[index] = distance.euclidean(
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
            distances[index] = distance.euclidean(self.__tmp_reduced_scaled[
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
                total_indexes=self.__total_indexes,)

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

        dp_1_dist = distance.euclidean(self.__tmp_reduced_scaled[dp_1_index],
                                       centroid)

        dp_2_dist = distance.euclidean(self.__tmp_reduced_scaled[dp_2_index],
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

    def __create_gif_dp_amount(self,
                               n_start,
                               n_end,
                               folder_dir_name,
                               filename,
                               flash_final_results=False,
                               show_gif=False):
        """
            Generates a gif based on pre-generated images of sample removal.
        """
        images = [imageio.imread(self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                                 folder_dir_name
                                 + "/Sample_removal_Visualized_Cluster_n={0}.png".format(
            i))
                  for i in range(n_start,
                                 n_end - 1,
                                 -1)]
        if flash_final_results:
            images += [imageio.imread(
                self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                folder_dir_name
                + "/Sample_removal_Visualized_Cluster_n={0}.png".format(
                    n_start)), imageio.imread(
                self.__PROJECT.PATH_TO_OUTPUT_FOLDER + "/" +
                folder_dir_name
                + "/Sample_removal_Visualized_Cluster_n={0}_White_Outed.png".format(
                    n_end))] * 4

        imageio.mimsave(self.__PROJECT.PATH_TO_OUTPUT_FOLDER +
                        "/" + folder_dir_name + "/{0}.gif".format(filename),
                        images,
                        duration=.68)

        if show_gif:
            from IPython.display import Image
            display(Image(filename=self.__PROJECT.PATH_TO_OUTPUT_FOLDER +
                                   "/" + folder_dir_name + "/{0}.gif".format(
                filename)))

    def __visualize_data_points(self,
                                centroid,
                                scaled_data,
                                noise_removal_dps,
                                similar_removal_dps,
                                new_sample_amount,
                                zscore_high,
                                weighted_dist_value,
                                annotate,
                                output_path,
                                title,
                                remove_noise,
                                remove_similar,
                                new_dp_meta_noise_removal=None,
                                new_dp_meta_similar_removal=None,
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
        plt.gcf().text(.91, 0.9, "Weight={0:.2f}, Zscore={1:.2f}".format(
            weighted_dist_value, zscore_high), fontsize=12)

        # Plot existing data points
        for i in range(0, scaled_data.shape[0]):
            pl.scatter(np.mean(scaled_data[i]),
                       distance.euclidean(scaled_data[i], centroid),
                       c="#0080FF",
                       marker='o',
                       label="Existing data point")

            # plt.annotate('{0}'.format(i), xy=(np.mean(scaled_data[i]),
            #            distance.euclidean(scaled_data[i], centroid)),
            #              xytext=(np.mean(scaled_data[i]),
            #                      distance.euclidean(scaled_data[i], centroid)))

        # Plot data points removed from noise removal
        noise_removal_last_index = len(noise_removal_dps) - 1
        for index, dp_index in enumerate(noise_removal_dps):

            if white_out_mode:
                pl.scatter(np.mean(self.__scaled[dp_index]),
                           distance.euclidean(self.__scaled[dp_index],
                                              centroid),
                           c="#ffffff",
                           marker='X',
                           alpha=0)
            else:
                pl.scatter(np.mean(self.__scaled[dp_index]),
                           distance.euclidean(self.__scaled[dp_index],
                                              centroid),
                           c="#00A572",
                           marker='X',
                           label="Noise Removal")

        if annotate and new_dp_meta_noise_removal:

            # ---
            dp = new_dp_meta_noise_removal[0]

            # Find the correct angle to have the text and annotated line match
            mean_of_dp = np.mean(dp)
            dp_centroid_dist = distance.euclidean(dp, centroid)

            dy = (0 - dp_centroid_dist)
            dx = (np.mean(centroid) - mean_of_dp)
            rotn = np.degrees(np.arctan2(dy, dx))
            trans_angle = plt.gca().transData.transform_angles(
                np.array((rotn,)), np.array((mean_of_dp,
                                             dp_centroid_dist)).reshape(
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
            plt.annotate(' ', xy=(mean_of_dp, dp_centroid_dist), xytext=(
                np.mean(centroid), 0),
                         rotation=trans_angle,
                         ha='center',
                         va='center',
                         rotation_mode='anchor',
                         arrowprops={'arrowstyle': '<->', 'shrinkA': .4,
                                     'shrinkB': 4.5}
                         )

            # Create text
            plt.annotate(spacing + 'zscore={0:.2f} , Dist:{1:.2f}\n'.format(
                new_dp_meta_noise_removal[1],
                new_dp_meta_noise_removal[2]),
                         xy=(mean_of_dp * .5, dp_centroid_dist * .5),
                         rotation_mode='anchor',
                         va='center',
                         ha='center',
                         rotation=trans_angle)

        for index, dp_index in enumerate(similar_removal_dps):
            if white_out_mode:
                pl.scatter(np.mean(self.__scaled[dp_index]),
                           distance.euclidean(self.__scaled[dp_index],
                                              centroid),
                           c="#ffffff",
                           marker='X',
                           alpha=0)
            else:
                pl.scatter(np.mean(self.__scaled[dp_index]),
                           distance.euclidean(self.__scaled[dp_index],
                                              centroid),
                           c="#8A2BE2",
                           marker='X',
                           label="Similar Removal")

        if annotate and new_dp_meta_similar_removal:
            # Create line
            plt.annotate(' ', xy=(np.mean(new_dp_meta_similar_removal[0]),
                                  distance.euclidean(
                                      new_dp_meta_similar_removal[0],
                                      centroid)),
                         xytext=(np.mean(new_dp_meta_similar_removal[1]),
                                 distance.euclidean(
                                     new_dp_meta_similar_removal[1],
                                     centroid)),
                         ha='center',
                         va='center',
                         rotation_mode='anchor',
                         arrowprops={'arrowstyle': '<->', 'shrinkA': .4,
                                     'shrinkB': 4.5}
                         )

        # Plot centroid
        pl.scatter(np.mean(centroid), 0,
                   c="r", marker="D",
                   label="Centroid")
        # Author: http://tinyurl.com/yxvd33t2
        # Removes all duplicated handles and labels of the labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),
                   loc='center left', bbox_to_anchor=(1, 0.92))

        if white_out_mode:
            self.__create_plt_png(output_path,
                                  "Sample_removal_Visualized_Cluster_n={0}_White_Outed".format(
                                      len(scaled_data)))

        else:
            self.__create_plt_png(output_path,
                                  "Sample_removal_Visualized_Cluster_n={0}".format(
                                      len(scaled_data)))

        if display_all_graphs:
            plt.show()
        else:
            if self.__scaled.shape[0] > scaled_data.shape[
                0] and not no_print_output:
                if new_dp_meta_noise_removal:
                    print("Scaled size is now {0}".format(
                        scaled_data.shape[
                            0]) + " and Z-Score of {0:.2f}".format(
                        new_dp_meta_noise_removal[1]))

                else:
                    print("Scaled size is now {0}".format(scaled_data.shape[0]
                                                          ))

        plt.close()

    # --- Misc
    # # I am this lazy yes...
    # def __vertical_spacing(self, spaces=1):
    #     for _ in range(0, spaces):
    #         print()

    # --- Getters/Setters
    def get_scaled_data(self):
        return copy.deepcopy(self.__scaled)



