# This code is hidden for a reason; it is very ugly code; but it gets the job done.
# All "Hidden" functions except check_create_metadata_of_dataframe
# should never be used/called anywhere else.

import os
import pandas as pd
import math
import copy
import numpy as np
from IPython.display import display
import json

from eflow.utils.pandas_utils import data_types_table
from eflow.utils.string_utils import convert_to_filename
from eflow.utils.sys_utils import convert_to_filename, check_create_dir_structure, \
    create_json_object_from_dict
from eflow import DataFrameTypes
from eflow.utils.string_utils import correct_directory_path
from eflow.utils.image_utils import df_to_image
from eflow._hidden.custom_exceptions import *

class DataFrameIdenity:
    """
        Attempts to get the "idenity" of a dataframe by extracting varying data
        of the dataframe. This will help ensure Analysis object's
        Will not ever be able to confirm with 100%
    """

    def __init__(self,
                 compare_shape=True,
                 compare_feature_names=True,
                 compare_feature_types=True,
                 compare_random_values=True):
        self.__compare_shape = copy.deepcopy(compare_shape)
        self.__compare_feature_names = copy.deepcopy(compare_feature_names)
        self.__compare_feature_types = copy.deepcopy(compare_feature_types)
        self.__compare_random_values = copy.deepcopy(compare_random_values)



    def __create_random_values_from_string(self,
                                           feature_name,
                                           hash_type):
        """
        feature_name:
            The given feature's string name

        hash_type:
            Numeric value to determine which

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
            DataFrameTypeHolder object.

        Returns/Desc:
            Creates a dict object of the feature types
        """
        feature_types = dict()
        feature_types["integer_features"] = sorted(df_features.get_integer_features())
        feature_types["float_features"] = sorted(df_features.get_float_features())
        feature_types["bool_features"] = sorted(df_features.get_bool_features())
        feature_types["categorical_features"] = sorted(df_features.get_categorical_features())
        feature_types["datetime_features"] = sorted(df_features.get_datetime_features())

        return feature_types

    def __create_random_values_dict(self,
                                    df,
                                    df_features):
        """
        df:
            Pandas dataframe object

        df_features:
            DataFrameTypeHolder object.

        Returns/Desc:
            Creates a dict of random values
        """

        feature_values = dict()
        random_indexes = set()
        float_features = set(df_features.get_float_features())

        for feature in sorted(df_features.get_all_features()):

            if feature in float_features:
                continue

            feature_values[feature] = dict()
            for hash_type in list(range(1, 11)):
                random_index = __create_random_values_from_string(feature,
                                                                  hash_type) % \
                               df.shape[0]
                size_of_indexes = len(random_indexes)
                if df.shape[0] > 10:
                    for i in list(range(0, df.shape[0] + 1)):
                        if random_index + i >= df.shape[0]:
                            random_indexes.add((random_index + i) - df.shape[0])
                        else:
                            random_indexes.add(random_index + i)

                        if size_of_indexes != len(random_indexes):
                            random_index += i
                            break
                else:
                    random_indexes.add(random_index)

                feature_values[feature][f'Random Value {hash_type}'] = dict()

                random_value = df[feature][random_index]

                if not random_value or pd.isnull(random_value):
                    feature_values[feature][
                        f'Random Value {hash_type}'] = "NaN"

                else:
                    feature_values[feature][
                        f'Random Value {hash_type}'] = str(random_value)

        return feature_values


    def __generate_dataframe_identity_dict(self,
                                           df,
                                           compare_shape,
                                           compare_feature_names,
                                           compare_feature_types,
                                           compare_random_values):
        """
            This identity will not be perfect for giving an exact
            identity to each dataframe passed to it.
            But good enough to catch most mis
        """
        df_features = DataFrameTypes(df,
                                     display_init=False,
                                     ignore_nulls=True)

        generated_any_check = False
        meta_dict = dict()
        if compare_shape:
            meta_dict["shape"] = df.shape
            generated_any_check = True

        if compare_feature_names:
            meta_dict["feature_names"] = sorted(df_features.get_all_features())
            generated_any_check = True

        if compare_feature_types:
            meta_dict["feature_types"] = self.__create_feature_types_dict(df_features)
            generated_any_check = True

        if compare_random_values:
            meta_dict["random_values"] = self.__create_random_values_dict(df,df_features)
            generated_any_check = True

        if generated_any_check == False:
            # raise
            pass

        return meta_dict

    def __create_dataframe_idenity_json_file(self,
                                             df,
                                             output_folder_path,
                                             compare_shape,
                                             compare_feature_names,
                                             compare_feature_types,
                                             compare_random_values):
        output_folder_path = correct_directory_path(output_folder_path)

        # shape_df = pd.DataFrame.from_dict({'Rows': [df.shape[0]],
        #                                    'Columns': [df.shape[1]]})
        # if notebook_mode:
        #     if display_visuals:
        #         display(shape_df)
        # else:
        #     if display_visuals:
        #         print(shape_df)
        #
        # dtypes_df = data_types_table(df)
        #
        # if notebook_mode:
        #     if display_visuals:
        #         display(shape_df)
        # else:
        #     if display_visuals:
        #         print(shape_df)
        #
        # df_to_image(dtypes_df,
        #             output_folder_path,
        #             "_Extras",
        #             convert_to_filename("Data Types Table"),
        #             show_index=True,
        #             format_float_pos=2)

        meta_dict = __generate_dataframe_identity_dict(df,
                                                       compare_shape,
                                                       compare_feature_names,
                                                       compare_feature_types,
                                                       compare_random_values)

        create_json_object_from_dict(meta_dict,
                                     correct_directory_path(output_folder_path + "_Extras"),
                                     "Meta Data Identity")

    def check_idenity(self,
                      df,
                      output_folder_path):
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

        FLOATS WILL BE IGNORED!!!!!!!!!
        """
        output_folder_path = correct_directory_path(output_folder_path)

        if not os.path.exists(output_folder_path + "_Extras"):
            check_create_dir_structure(output_folder_path,
                                       "_Extras")


        json_file = output_folder_path + "_Extras/Dataframe Identity.json"

        # Meta Data has already been generated
        if os.path.isfile(json_file):
            with open(json_file) as file:
                data = json.load(file)

                if not compare_shape and not compare_feature_names and not compare_feature_types and not compare_random_values:
                    raise

                df_features = DataFrameTypes(df,
                                             display_init=False,
                                             ignore_nulls=True)
                mismatch_error = None
                while True:
                    if compare_shape:
                        if list(data["shape"]) != list(df.shape):
                            mismatch_error = f'Saved shape {data["shape"]} of the' \
                                             f' dataframe did not match up with' \
                                             f' the passed dataframe shape {df.shape}.'
                            break

                    if compare_feature_names:
                        if data["feature_names"] != sorted(df_features.get_all_features()):
                            mismatch_error = f'Saved features {data["feature_names"]} of the' \
                                             f' dataframe did not match up with' \
                                             f' the passed dataframe features {sorted(df_features.get_all_features())}.'
                            break

                    if compare_feature_types:
                        compared_data = self.__create_feature_types_dict(df_features)
                        if data["feature_types"] != compared_data:
                            mismatch_error = f'Saved features {data["feature_types"]} ' \
                                f'of the dataframe did not match up with' \
                                f' the passed dataframe features {compared_data}.'
                            break



                    if compare_random_values:
                        compared_data = self.__create_random_values_dict(df,
                                                                         df_features)
                        print(compared_data)
                        print("\n\n\n\n")
                        print(data["random_values"])
                        if data["random_values"] != compared_data:
                            mismatch_error = "Random values did not match at the proper places in the dataframe."
                            break

                    break

                if mismatch_error:
                    raise ValueNotAsExpected(f"DataFrameIdenity has raised an error because {mismatch_error}.\n" \
                                             f"This error can be resolved by:" \
                                             f"\n\t* Pass in the same dataframe as expected."
                                             f"\n\t* Disable the identity check by changing 'idenity_check' to False."
                                             f"\n\t* Disable the ")


        else:
            __create_dataframe_idenity_json_file(df,
                                                 output_folder_path,
                                                 compare_shape=compare_shape,
                                                 compare_feature_names=compare_feature_names,
                                                 compare_feature_types=compare_feature_types,
                                                 compare_random_values=compare_random_values)