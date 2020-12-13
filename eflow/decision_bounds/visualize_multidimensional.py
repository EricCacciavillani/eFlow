
from eflow._hidden.parent_objects import AutoModeler
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow.utils.sys_utils import load_pickle_object,get_all_files_from_path, get_all_directories_from_path, pickle_object_to_file, create_dir_structure, write_object_text_to_file, json_file_to_dict, dict_to_json_file
from eflow.utils.eflow_utils import move_folder_to_eflow_garbage
from eflow.utils.math_utils import euclidean_distance

# Getting Sklearn Models
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Visuals libs
import matplotlib.pyplot as plt
from IPython.display import display, HTML

# Misc
import pandas as pd
import numpy as np
import copy
import math
import os
from tqdm import tqdm


from eflow._hidden.helper_functions.visualize_multidimensional_multi_threading import find_all_distances_in_matrix

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class VisualizeMultidimensional(AutoModeler):
    """
        Analyzes the feature data of a pandas Dataframe object.
    """

    def __init__(self,
                 df,
                 feature_names=[],
                 dataset_sub_dir="",
                 dataset_name="Default Dataset Name",
                 overwrite_full_path=None,
                 notebook_mode=False,
                 pca_perc=1.00):
        """
        Args:
            df: pd.Dataframe
                pd.Dataframe

            dataset_sub_dir: string
                Sub directory to write data.

            dataset_name: string
                Main project directory

            overwrite_full_path: string
                Overwrite full directory path to a given output folder

            notebook_mode: bool
                Display and show in notebook if set to true.
        """

        if isinstance(df, pd.DataFrame):
            self.__feature_names = copy.deepcopy(list(df.columns))
        else:
            if not feature_names:
                raise UnsatisfiedRequirments("If passing in a matrix like object. "
                                             "You must init feature names!")
            else:
                self.__feature_names = copy.deepcopy(feature_names)


        AutoModeler.__init__(self,
                             f'{dataset_name}/{dataset_sub_dir}',
                             overwrite_full_path)

        # Define model
        self.__cluster_models_paths = dict()

        self.__notebook_mode = copy.deepcopy(notebook_mode)

        self.__models_suggested_clusters = dict()

        self.__pca = None

        self.__first_scaler = None
        self.__second_scaler = None
        self.__cutoff_index = None
        self.__ordered_dp_indexes = None
        self.__pca_perc = pca_perc

        # --- Apply pca ---
        if pca_perc:

            # Create scaler object
            scaler = StandardScaler()
            scaled = scaler.fit_transform(df)

            self.__first_scaler = copy.deepcopy(scaler)

            print("\nInspecting scaled results!")
            self.__inspect_feature_matrix(matrix=scaled,
                                          feature_names=self.__feature_names,
                                          sub_dir="PCA",
                                          filename="Applied scaler results")

            pca, scaled = self.__visualize_pca_variance(scaled)

            self.__pca = pca

            # Generate "dummy" feature names
            pca_feature_names = ["PCA_Feature_" +
                                 str(i) for i in range(1,
                                                       len(self.__feature_names) + 1)]

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

        # Save objects to directory structure
        if self.__pca:
            pipeline_path = create_dir_structure(self.folder_path,
                                                 "Data Cluster Pipeline")

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


    def sort_data_points_by_distances(self):
        all_dp_distances = find_all_distances_in_matrix(self.__scaled)


        # print(all_dp_distances[0])
        shifting_dp_index = 74
        ordered_dp_indexes = [shifting_dp_index]
        for _ in range(0, len(all_dp_distances.keys())):

            closeset_dp_index = None
            closeset_dp_distance = float("inf")

            # All data point indexes below the shifting index
            for dp_i in range(0,shifting_dp_index):
                if dp_i in all_dp_distances.keys():

                    if closeset_dp_distance > all_dp_distances[dp_i][shifting_dp_index - dp_i - 1]:
                        closeset_dp_index = dp_i
                        closeset_dp_distance = all_dp_distances[dp_i][shifting_dp_index - dp_i - 1]


                    all_dp_distances[dp_i][
                        shifting_dp_index - dp_i - 1] = np.nan

            # All data point indexes above the shifting index; ignore the last dp index
            if self.__scaled.shape[0] - 1 != shifting_dp_index:

                tmp_index = None
                try:
                    tmp_index = np.nanargmin(all_dp_distances[shifting_dp_index])

                except ValueError:
                    pass


                if tmp_index and closeset_dp_distance > all_dp_distances[shifting_dp_index][tmp_index]:
                    closeset_dp_index = tmp_index + shifting_dp_index + 1

                del all_dp_distances[shifting_dp_index]

            ordered_dp_indexes.append(closeset_dp_index)
            del closeset_dp_distance

            shifting_dp_index = closeset_dp_index

            # for dp_i in range(0,shifting_dp_index):
            #     if dp_i in dp_keys:
            #         distances.append(all_dp_distances[dp_i][shifting_dp_index-dp_i])
            #         all_dp_distances[dp_i][shifting_dp_index-dp_i] = np.nan
            #         print(shifting_dp_index-dp_i)

            #
            # print(f"shifting_dp_index={shifting_dp_index}")
            # print(f"len = {len(distances)}")
            # print("\n\n\n")
            # distances = np.concatenate((np.array(distances), all_dp_distances[shifting_dp_index]))
            # del all_dp_distances[shifting_dp_index]
            #
            # indexes = np.argsort(distances)
            #
            # for i in indexes:
            #
            #     if not math.isnan(distances[i]):
            #
            #         ordered_dp_indexes.append(i)
            #         shifting_dp_index = i
            #         break
            #
            #     else:
            #         print("Oh dear god all of math is dead! God himself hates you! Die kys")


        print(ordered_dp_indexes)
        print(len(ordered_dp_indexes))

        self.__ordered_dp_indexes = ordered_dp_indexes


    # --- Getters/Setters
    def get_ordered_dp_indexes(self):
        """

            Get a copy of the stored data

        Returns:
            Returns the stored data
        """
        return copy.deepcopy(self.__ordered_dp_indexes)

    def get_scaled_data(self):
        """

            Get a copy of the stored data

        Returns:
            Returns the stored data
        """
        return copy.deepcopy(self.__scaled)

    def delete_scaled_data(self):
        """
            Removes the matrix data in order to save RAM when running
            analysis on the actual data.
        """
        del self.__scaled
        self.__scaled = None

    def apply_clustering_data_pipeline(self,
                                       data):
        """
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


    def __inspect_feature_matrix(self,
                                 matrix,
                                 feature_names,
                                 sub_dir,
                                 filename):
        """

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