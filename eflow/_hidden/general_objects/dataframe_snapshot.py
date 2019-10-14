import os
import pandas as pd
import math
from eflow.foundation import DataFrameTypes
from eflow.utils.sys_utils import check_create_dir_structure, \
    create_json_file_from_dict

from eflow.utils.string_utils import correct_directory_path
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, MismatchError
import copy
import json
import numpy as np


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
                 compare_feature_types=True,
                 compare_random_values=True):
        """

        compare_shape:
            Determines whether or not to create/compare the dataframe's shape for the snapshot

        compare_feature_names:
            Determines whether or not to create/compare the dataframe's the feature names for the snapshot

        compare_feature_types:
            Determines whether or not to create/compare the dataframe's the feature types for the snapshot

        compare_random_values:
            Determines whether or not to create/compare the dataframe's 10 'random'
            values found on each feature. As long as the same dataframe is passed the random values should be the same.
            Note:
                Will ignore float features because of trailing value problem that all floats have.
        """

        # Copy values
        self.__compare_shape = copy.deepcopy(compare_shape)
        self.__compare_feature_names = copy.deepcopy(compare_feature_names)
        self.__compare_feature_types = copy.deepcopy(compare_feature_types)
        self.__compare_random_values = copy.deepcopy(compare_random_values)

        # Error check; must have at least one boolean
        if not self.__compare_shape and \
                not self.__compare_feature_names and \
                not self.__compare_feature_types and \
                not self.__compare_random_values:
            raise UnsatisfiedRequirments("At least one compare boolean must be "
                                         "set to True for snapshot check to properly work")

    def __create_random_values_from_string(self,
                                           feature_name,
                                           hash_type):
        """
        feature_name:
            The given feature's string name.

        hash_type:
            Numeric value to determine which.

        Returns/Desc:
            Returns back a value based on the string and the hash_type number
            passed to it.
        """

        if feature_name == "0":
            feature_name = "1"

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
        result = np.absolute(result)

        # return np.absolute(np.absolute(hash(feature_name)) - result)

        return result

    def __create_feature_types_dict(self,
                                    df_features):
        """
        df_features:
            DataFrameTypes object; organizes feature types into groups.

        Returns/Desc:
            Creates a dict object of the feature types
        """
        feature_types = dict()
        feature_types["integer_features"] = sorted(list(df_features.get_integer_features()))
        feature_types["float_features"] = sorted(list(df_features.get_float_features()))
        feature_types["bool_features"] = sorted(list(df_features.get_bool_features()))
        feature_types["string_features"] = sorted(list(df_features.get_string_features()))
        feature_types["datetime_features"] = sorted(list(df_features.get_datetime_features()))

        return feature_types

    def __create_random_values_dict(self,
                                    df,
                                    df_features):
        """
        df:
            Pandas dataframe object

        df_features:
            DataFrameTypes object; organizes feature types into groups.

        Returns/Desc:
            Creates a dict of 10 random values for each value based on the
            feature's name.
        """

        feature_values = dict()
        random_indexes = set()
        float_features = set(df_features.get_float_features())

        for feature in sorted(df_features.get_all_features()):

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
                random_value = df[feature][random_index]

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
        df:
            Pandas dataframe object

        Returns/Desc:
            Returns back a dict representing the snapshot of the dataframe.

            Note:
                Uses the following private variables to decide whether or not to generate
                that component of the dataframe.
        """
        df_features = DataFrameTypes(df,
                                     display_init=False,
                                     ignore_nulls=True)

        meta_dict = dict()
        if self.__compare_shape:
            meta_dict["shape"] = list(df.shape)

        if self.__compare_feature_names:
            meta_dict["feature_names"] = sorted(df_features.get_all_features())

        if self.__compare_feature_types:
            meta_dict["feature_types"] = self.__create_feature_types_dict(df_features)

        if self.__compare_random_values:
            meta_dict["random_values"] = self.__create_random_values_dict(df,df_features)

        return meta_dict

    def __create_dataframe_snapshot_json_file(self,
                                             df,
                                             output_folder_path):
        """

        df:
            Pandas Dataframe object

        output_folder_path:
            Output path the json object will move to.

        Returns/Desc:
            Creates a json file based on the dataframe's generated snapshot dict.
        """
        output_folder_path = correct_directory_path(output_folder_path)

        meta_dict = self.__generate_dataframe_snapshot_dict(df)

        create_json_file_from_dict(meta_dict,
                                     output_folder_path,
                                     "Dataframe Identity")

    def check_create_snapshot(self,
                              df,
                              directory_pth,
                              sub_dir):
        """
        df:
            Pandas dataframe object.

        output_folder_path:
            Output path of the dataset's.

        display_visuals:
            If set to True than it will visualize the given data.

        notebook_mode:
            If set to True than will attempt to visualize the data in a notebook if
            'display_visuals' is set to True.

        Returns/Desc:

        """

        output_folder_path = check_create_dir_structure(directory_pth,
                                                        sub_dir)

        json_file = output_folder_path + "Dataframe Identity.json"

        # Meta Data has already been generated; compare data
        if os.path.isfile(json_file):
            with open(json_file) as file:

                data = json.load(file)

                df_features = DataFrameTypes(df,
                                             display_init=False,
                                             ignore_nulls=True)
                mismatch_error = None
                while True:
                    if self.__compare_shape:
                        list_shape = list(df.shape)
                        if list(data["shape"]) != list_shape:
                            mismatch_error = f'the saved shape {data["shape"]} of the' \
                                             f' dataframe snapshot did not match up with' \
                                             f' the passed dataframe shape {list_shape}.'
                            break

                    if self.__compare_feature_names:

                        snapshot_features = set(data["feature_names"])
                        passed_features = set(df_features.get_all_features())

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

                    if self.__compare_feature_types:
                        passed_feature_types_dict = self.__create_feature_types_dict(df_features)

                        # ----------
                        for snapshot_feature_type,snapshot_feature_list in data["feature_types"].items():

                            # --------
                            for snapshot_feature in snapshot_feature_list:

                                # Mismatch found between snapshot and passed dataframe
                                if snapshot_feature not in passed_feature_types_dict[snapshot_feature_type]:

                                    # Error labeling
                                    correct_feature_type = "UNKNOWN TYPE"
                                    if mismatch_error is None:
                                        mismatch_error = ":"

                                    # Find the correct type of the feature
                                    for passed_feature_type, passed_feature_list in passed_feature_types_dict.items():

                                        # We already confirmed that it isn't this data type
                                        if passed_feature_type == snapshot_feature_type:
                                            continue
                                        if snapshot_feature in passed_feature_list:
                                            correct_feature_type = passed_feature_type
                                            break

                                    # Add to error message
                                    mismatch_error += f"\n\tFeature '{snapshot_feature}' is supposed " \
                                        f"to be in '{snapshot_feature_type}' was found to be in '{correct_feature_type}'"

                        break


                    if self.__compare_random_values:
                        compared_data = self.__create_random_values_dict(df,
                                                                         df_features)
                        if data["random_values"] != compared_data:
                            mismatch_error = "the 'random' values did not match at the proper places in the dataframe " \
                                             "(these 'random' values are based on the shape and name of the column)."
                            break

                    break
                if mismatch_error is not None:
                    raise MismatchError(f"DataFrameSnapshot has raised an error because {mismatch_error}." +
                                        "\nThis error invoked because the directory structure saved a json file "
                                        "containing attributes of the dataframe or a 'snapshot'."
                                        "\nThe given error can be resolved by performing any of the following:"
                                        "\n\t* Pass in the same dataframe as expected."
                                        "\n\t* Disable the identity check by changing 'dataframe_snapshot' to False."
                                        "\n\t* Disable save file option by changing the parameter 'save_file' to False."
                                        "\n\t* Or deleting the json object file in the dataset directory under _Extras")

        # JSON file doesn't exist; create file
        else:
            self.__create_dataframe_snapshot_json_file(df,
                                                       output_folder_path)