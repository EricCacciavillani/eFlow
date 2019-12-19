from eflow.utils.string_utils import correct_directory_path
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, SnapshotMismatchError
from eflow.foundation import DataFrameTypes
from eflow.utils.sys_utils import create_dir_structure, \
    dict_to_json_file

import os
import pandas as pd
import math
import copy
import json
import numpy as np

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class DataFrameSnapshot:
    """
        Attempts to get a "snapshot" of a dataframe by extracting varying data
        of the pandas dataframe object; then generates a file to later compare
        in a set directory. Helps ensures that a dataset's directory structure
        only ever analyzes/works with that given dataframe.

        Shorthand:
            1: You generate a graph with dataset A and place it in a directory called A's Data.
            2: You generate a graph with dataset B and try to place it in A's Data.

            An error should invoke; thus protecting against creating file's
            that are relating to differing dataframes.

        Note:
            Every function using this object has the ability will have the
            ability to turn on/off this generation and checking.
            'dataframe_snapshot'
    """

    def __init__(self,
                 compare_shape=True,
                 compare_feature_names=True,
                 compare_random_values=True):
        """
        Args:
            compare_shape: bool
                Determines whether or not to create/compare the dataframe's shape for the snapshot

            compare_feature_names: bool
                Determines whether or not to create/compare the dataframe's the feature names for the snapshot


            compare_random_values: bool
                Determines whether or not to create/compare the dataframe's 10 'random'
                values found on each feature. As long as the same dataframe is passed the random values should be the same.
                Note:
                    Will ignore float features because of trailing value problem that all floats have.
        """

        # Copy values
        self.__compare_shape = copy.deepcopy(compare_shape)
        self.__compare_feature_names = copy.deepcopy(compare_feature_names)
        self.__compare_random_values = copy.deepcopy(compare_random_values)

        # Error check; must have at least one boolean
        if not self.__compare_shape and \
                not self.__compare_feature_names and \
                not self.__compare_random_values:
            raise UnsatisfiedRequirments("At least one compare boolean must be "
                                         "set to True for snapshot check to properly work")

    def check_create_snapshot(self,
                              df,
                              df_features,
                              directory_path,
                              sub_dir):
        """
        Desc:
            Compares the passed pandas dataframe object to pre defined json
            file.

        Args:
            df: pd.Dataframe
                Pandas dataframe object.

            directory_path: string
                Output path of the dataset's.

            sub_dir: string
                If set to True than it will visualize the given data.

        Raises:
            Will raise a Mismatch error if the json file didn't match upp with the
            passed dataframe snapshot; causing the program to stop in runtime.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"'df' must be a pandas datafram object not a {type(df)}")

        if not isinstance(df_features,DataFrameTypes):
            raise TypeError(f"'df_features' must be a DataFrameTypes object not a {type(df_features)}")


        output_folder_path = create_dir_structure(directory_path,
                                                  sub_dir)

        json_file = output_folder_path + "Dataframe Snapshot.json"

        # Meta Data has already been generated; compare data
        if os.path.isfile(json_file):
            with open(json_file) as file:

                data = json.load(file)
                mismatch_error = None

                # Not using for looping; used for logic breaks
                while True:
                    if self.__compare_shape:
                        list_shape = list(df.shape)
                        if list(data["shape"]) != list_shape:
                            mismatch_error = f'the saved shape {data["shape"]} of the' \
                                             f' dataframe snapshot did not match up with' \
                                             f' the passed dataframe shape {list_shape}.'
                            break

                    # Ensure feature names match up
                    if self.__compare_feature_names:

                        snapshot_features = set(data["feature_names"])
                        passed_features = set(df_features.all_features())

                        feature_difference = snapshot_features.symmetric_difference(
                            passed_features)

                        if feature_difference:
                            mismatch_error = "the following feature name conflicts feature:\n"

                            missing_features = []
                            for feature in feature_difference:
                                if feature in snapshot_features:
                                    missing_features.append(feature)

                            extra_features = []
                            for feature in feature_difference:
                                if feature in passed_features:
                                    extra_features.append(feature)

                            if extra_features:
                                mismatch_error += f"--- Passed dataframe has additional feature(s) than snapshot:\n {extra_features}.\n"

                            if missing_features:
                                mismatch_error += f"--- Passed dataframe is missing the following snapshot feature(s):\n {missing_features}.\n"

                            if extra_features or missing_features:
                                break

                    # Ensure sudo random numbers are chosen again
                    if self.__compare_random_values:
                        compared_data = self.__create_random_values_dict(df,
                                                                         df_features)

                        random_values_matchd_flag = True
                        for k,v in data["random_values"].items():
                            if k in compared_data:
                                if data["random_values"][k] != compared_data[k]:
                                    random_values_matchd_flag = False
                                    break
                        if not random_values_matchd_flag:
                            mismatch_error = f"the 'random' values did not match at feature name '{k}' in the dataframe " \
                                             + "(these 'random' values are based on the shape and name of the column)"
                            break

                    # Break main loop
                    break

                # Error found; raise it
                if mismatch_error is not None:
                    raise SnapshotMismatchError(f"DataFrameSnapshot has raised an error because {mismatch_error}." +
                                                "\nThis error invoked because the directory structure saved a json file "
                                                "containing attributes of the dataframe or a 'snapshot'."
                                                "\nThe given error can be resolved by performing any of the following:"
                                                "\n\t* Pass in the same dataframe as expected."
                                                "\n\t* Disable the snapshot check by changing 'dataframe_snapshot' to False."
                                                "\n\t* Disable save file option by changing the parameter 'save_file' to False."
                                                "\n\t* Or deleting the json object file in the dataset directory under _Extras")

        # JSON file doesn't exist; create file
        else:
            self.__create_dataframe_snapshot_json_file(df,
                                                       output_folder_path)


    def __create_random_values_from_string(self,
                                           feature_name,
                                           hash_type):
        """
        Desc:
            Create a sudo random integer based off the name of the feature and
            a number between 1 and 10.

        Args:
            feature_name: string
                The given feature's string name.

            hash_type: int (1-10)
                Numeric value to determine which.

        Returns:
            Returns back a sudo random value based on arguments.
        """

        # Apply each char to a calculation
        result = 0
        for char_index, char in enumerate(feature_name):
            char = str(char)
            if hash_type == 1:
                result += int(ord(char))
            elif hash_type == 2:
                result += int(ord(char) + 62 * ord(char))
            elif hash_type == 3:
                result += int(ord(char) + 147 * ord(char))
            elif hash_type == 4:
                result += int((ord(char) + 92) * math.pow(ord(char), 3))
            elif hash_type == 5:
                result += int(ord(char) + 49 * math.pow(ord(char), 2))
            elif hash_type == 6:
                result += int((23 + ord(char) + 45) * (3 + ord(char) * 2))
            elif hash_type == 7:
                result += int((ord(char) * 5) + 32 + 8)
            elif hash_type == 8:
                result += int(math.pow(ord(char), 2))
            elif hash_type == 9:
                result += int(ord(char) * 2 + 32 + ord(char) * 2 + 5)
            elif hash_type == 10:
                result += int(ord(char) * 4 * 12 + 76 + math.pow(ord(char), 2))
            else:
                raise ValueError("Hash type must be between 1-10.")

        # Make sure the number is positive (Future proofing if someone adds in
        # something that could cause the result to be negative.)
        result = np.absolute(result)

        return result


    def __create_random_values_dict(self,
                                    df,
                                    df_features):
        """
        Desc:
            Creates a dict of 10 random values for each value based on the
            feature's name and the count of the random values already
            generated.

        Args:
            df: pd.Dataframe
                Pandas dataframe object

            df_features: DataFrameTypes from eflow
                DataFrameTypes object; organizes feature types into groups.

        Returns:
            Returns back a dict of 10 random values based on the arguments.
        """

        feature_values = dict()
        random_indexes = set()
        float_features = set(df_features.float_features())

        # Empty dataframe check
        if not df.shape[0]:
            return feature_values

        for feature in sorted(df_features.all_features()):

            # Ignore if feature is a float
            if feature in float_features:
                continue

            # Random values based on unique indexes for a given feature dict
            feature_values[feature] = dict()
            for hash_type in list(range(1, 11)):
                random_index = self.__create_random_values_from_string(feature,
                                                                       hash_type) % \
                               df.shape[0]
                size_of_indexes = len(random_indexes)

                # Feature needs at least 10 indexes for 10 unique indexes
                if df.shape[0] > 10:
                    # Keep changing the 'random_index' until the value is unique within the set
                    for i in list(range(0, df.shape[0])):

                        random_index += i
                        if random_index >= df.shape[0]:
                            random_index -= df.shape[0]

                        if size_of_indexes != len(random_indexes):
                            break
                else:
                    random_indexes.add(random_index)

                # --------
                feature_values[feature][f'Random Value {hash_type}'] = dict()
                random_value = df[feature].iloc[random_index]

                # Random value is nan
                if not random_value or pd.isnull(random_value):
                    feature_values[feature][
                        f'Random Value {hash_type}'] = "NaN"

                # Convert the value to string no matter what; (Almost all objects have a str representation)
                else:
                    feature_values[feature][
                        f'Random Value {hash_type}'] = str(random_value)

        return feature_values


    def __generate_dataframe_snapshot_dict(self,
                                           df):
        """
        Desc:
            Creates a dict representing a snapshot of the dataframe based on
            the arguments when the object was first inited.

        Args:
            df: pd.Dataframe
                Pandas dataframe object

        Returns:
            Returns back a dict representing the snapshot of the dataframe.

            Note:
                Uses the following private variables to decide whether or not to generate
                that component of the dataframe.
        """
        df_features = DataFrameTypes(df,
                                     ignore_nulls=True)

        meta_dict = dict()
        if self.__compare_shape:
            meta_dict["shape"] = list(df.shape)

        if self.__compare_feature_names:
            meta_dict["feature_names"] = sorted(df_features.all_features())

        if self.__compare_random_values:
            meta_dict["random_values"] = self.__create_random_values_dict(df,
                                                                          df_features)

        return meta_dict


    def __create_dataframe_snapshot_json_file(self,
                                              df,
                                              output_folder_path):
        """
        Desc:
            Creates a json file based on the dataframe's generated snapshot dict.

        Args:
            df: pd.Dataframe
                Pandas Dataframe object

            output_folder_path: string
                Output path the json object will move to.
        """
        output_folder_path = correct_directory_path(output_folder_path)

        meta_dict = self.__generate_dataframe_snapshot_dict(df)

        dict_to_json_file(meta_dict,
                                   output_folder_path,
                                   "Dataframe Snapshot")

