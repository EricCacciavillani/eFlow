from eflow.utils.sys_utils import dict_to_json_file,json_file_to_dict
from eflow.utils.language_processing_utils import get_synonyms
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow._hidden.constants import BOOL_STRINGS

import copy

import numpy as np
import pandas as pd
from dateutil import parser
from IPython.display import display
from IPython.display import clear_output

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class DataFrameTypes:

    """
        Separates the features based off of dtypes to better keep track of
        feature types and helps make type assertions.
    """

    def __init__(self,
                 df=None,
                 target_feature=None,
                 ignore_nulls=False,
                 fix_numeric_features=False,
                 fix_string_features=False,
                 notebook_mode=False):
        """
        Args:
            df: pd.DataFrame
                Pandas dataframe object.

            target_feature: string
                If the project is using a supervised learning approach we can
                specify the target column. (Note: Not required)

            ignore_nulls: bool
                If set to true than a temporary dataframe is created with each
                feature removes it's nan values to assert what data type the series
                object would be without nans existing inside it.

            fix_numeric_features: bool
                Will attempt to convert all numeric features to the most proper
                numerical types.

            fix_string_features: bool
                Will attempt to convert all string features to ALL proper types.

            notebook_mode: bool
                Boolean value to determine if any notebook functions can be used here.
        """

        # Init an empty dataframe
        if df is None:
            df = pd.DataFrame({})

        # Grab features based on there types
        self.__bool_features = set(
            df.select_dtypes(include=["bool"]).columns)
        self.__string_features = set(
            df.select_dtypes(include=["object"]).columns)
        self.__categorical_features = set(
            df.select_dtypes(include=["category"]).columns)
        self.__integer_features = set(
            df.select_dtypes(include=["int"]).columns)
        self.__float_features = set(
            df.select_dtypes(include=["float"]).columns)
        self.__datetime_features = set(
            df.select_dtypes(include=["datetime"]).columns)

        null_features = df.isnull().sum()
        null_features[null_features == df.shape[0]].index.to_list()
        self.__null_only_features = set(null_features[null_features == df.shape[0]].index.to_list())

        del null_features

        # Target feature for machine learning projects
        self.__target_feature = None

        # Feature's colors
        self.__feature_value_color_dict = dict()

        # Category/Label encoders
        self.__label_encoder = dict()
        self.__label_decoder = dict()

        # Feature values representation
        self.__feature_value_representation = dict()

        # Dummy encoded feature dictionaries
        self.__dummy_encoded_features = dict()

        # Feature's labels and bins
        self.__feature_labels_bins_dict = dict()

        # Data type assertions without nulls
        if ignore_nulls and df.isnull().values.any():
            self.fix_nan_features(df)

        # Attempt to init target column
        if target_feature:
            if target_feature in df.columns:
                self.__target_feature = target_feature
            else:
                raise KeyError(f"The given target feature: \'{target_feature}\' does not exist!")

        # Error checking; flag error; don't disrupt runtime
        features_not_captured = set(df.columns)
        all_features = (self.__float_features | self.__integer_features) | \
                        self.__string_features | \
                        self.__bool_features | \
                        self.__datetime_features | \
                        self.__categorical_features | \
                        self.__null_only_features

        for col_feature in all_features:
            features_not_captured.remove(col_feature)

        if features_not_captured:
            print("ERROR UNKNOWN FEATURE(S) TYPE(S) FOUND!\n{0}".format(
                features_not_captured))

        if fix_string_features:
            self.fix_string_features(df,
                                     notebook_mode)

        if fix_numeric_features:
            self.fix_numeric_features(df,
                                      notebook_mode)

    # --- Getters ---
    def numerical_features(self,
                           exclude_target=False):
        """
        Desc:
            Gets all numerical features chosen by the object.

        Args:
            exclude_target: bool
                If the target feature is an numerical (int/float/bool); then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all numerical features chosen by the object.
        """
        tmp_set = self.__float_features | self.__integer_features | self.__bool_features
        if exclude_target:

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return tmp_set

    def non_numerical_features(self,
                               exclude_target=False):
        """
        Desc:
            Gets all non-numerical features chosen by the object.

        Args:
            exclude_target: bool
                If the target feature is an numerical (int/float/bool); then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all non numerical features chosen by the object.
        """
        tmp_set = self.all_features() ^ self.numerical_features()
        if exclude_target:

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return tmp_set

    def continuous_numerical_features(self,
                                      exclude_target=False):
        """
        Desc:
            Gets all numerical features that are continuous (int/float).

        Args:
            exclude_target: bool
                If the target feature is an numerical (int/float); then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all numerical features chosen by the object.
        """
        tmp_set = self.__float_features | self.__integer_features
        if exclude_target:

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return tmp_set

    def non_continuous_numerical_features(self,
                                          exclude_target=False):
        """
        Desc:
            Gets all numerical features that are not continuous (bool)

        Args:
            exclude_target: bool
                If the target feature is a bool; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all numerical features chosen by the object.
        """
        tmp_set = self.__bool_features
        if exclude_target:

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return tmp_set

    def continuous_features(self,
                            exclude_target=False):
        """
        Desc:
            Gets all numerical features chosen by the object.

        Args:
            exclude_target: bool
                If the target feature is an numerical (int/float/time); then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all numerical features chosen by the object.
        """
        tmp_set = self.__float_features | self.__integer_features | self.__datetime_features
        if exclude_target:

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return tmp_set

    def non_continuous_features(self,
                                exclude_target=False):
        """
        Desc:
            Gets all numerical features chosen by the object.

        Args:
            exclude_target: bool
                If the target feature is an numerical (int/float/time); then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all numerical features chosen by the object.
        """
        tmp_set = self.all_features() ^ self.continuous_features()
        if exclude_target:

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return tmp_set

    def integer_features(self,
                         exclude_target=False):
        """
        Desc:
            All integer features chosen by df_features.

        Args:
            exclude_target: bool
                If the target feature is an integer; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all integer features chosen by the object.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__integer_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__integer_features)

    def float_features(self,
                           exclude_target=False):
        """
        Desc:
            All float features chosen by df_features.

        Args:
            exclude_target: bool
                If the target feature is an float; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all float features chosen by the object.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__float_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__float_features)

    def categorical_features(self,
                             exclude_target=False):
        """
        Desc:
            All categorical features chosen by df_features.

        Args:
            exclude_target: bool
                If the target feature is an categorical; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all categorical features chosen by the object.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__categorical_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__categorical_features)

    def string_features(self,
                        exclude_target=False):
        """
        Desc:
            All string features chosen by df_features.

        Args:
            exclude_target: bool
                If the target feature is an string; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all string features chosen by the object.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__string_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__string_features)

    def bool_features(self,
                      exclude_target=False):

        """
        Desc:
            All bool features chosen by df_features.

        Args:
            exclude_target: bool
                If the target feature is an bool; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all bool features chosen by the object.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__bool_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__bool_features)

    def datetime_features(self,
                          exclude_target=False):
        """
        Desc:
            All datetime features chosen by df_features.

        Args:
            exclude_target: bool
                If the target feature is an datetime; then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all datetime features chosen by the object.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__datetime_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__datetime_features)


    def null_only_features(self,
                           exclude_target=False):


        if exclude_target:
            tmp_set = copy.deepcopy(self.__null_only_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__null_only_features)

    def all_features(self,
                     exclude_target=False):
        """
        Desc:
            Returns all features found in the dataset.

        Args:
            exclude_target: bool
                If the target feature is an datetime; then it will be ignored
                when passing back the set.
        Returns:
            Returns all found features.
        """

        all_features = (self.__float_features | self.__integer_features) | \
                        self.__string_features | \
                        self.__bool_features | \
                        self.__datetime_features | \
                        self.__categorical_features | \
                        self.__null_only_features

        if exclude_target:
            tmp_set = copy.deepcopy(all_features)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(all_features)


    def get_feature_type(self,
                         feature_name):
        """
        Desc:
            Return's a feature's type as a string

        Args:
            feature_name: str
                The given's feature's name.

        Returns:
            Return's a feature's type as a string
        """

        if feature_name in self.__float_features:
            return "float"

        elif feature_name in self.__bool_features:
            return "bool"

        elif feature_name in self.__integer_features:
            return "integer"

        elif feature_name in self.__categorical_features:
            return "categorical"

        elif feature_name in self.__string_features:
            return "string"

        elif feature_name in self.__datetime_features:
            return "datetime"

        elif feature_name in self.__null_only_features:
            return "null only"

        else:
            raise KeyError(f"Feature '{feature_name}' can't be found in the set!")

    def target_feature(self):
        """
        Desc:
            Gets the target feature.

        Returns:
            Returns the target feature.
        """
        return copy.deepcopy(self.__target_feature)


    def get_label_decoder(self):
        """
        Desc:
            Gets the dict encoder for category to string relationship.

        Returns:
            Returns the encoder dict object.
        """
        return copy.deepcopy(self.__label_decoder)

    def get_label_encoder(self):
        """
        Desc:
            Gets the dict encoder for string to category relationship.

        Returns:
            Returns the encoder dict object.
        """
        return copy.deepcopy(self.__label_encoder)

    def get_feature_colors(self,
                           feature_name):
        """
        Desc:
            Get's the color values for that given feature.

        Args:
            feature_name: str
                The given feature name of the

        Returns:
            Returns the value's color and dictionary; returns None if the feature name
            is not saved in the feature value color dict.
        """
        if feature_name in self.__feature_value_color_dict.keys():
            return copy.deepcopy(self.__feature_value_color_dict[feature_name])
        else:
            return None


    def get_all_feature_colors(self):
        """
        Desc:
           Get's the entire dict of the feature's value colors.

        Returns:
            Returns a copy of feature's value colors.
        """
        return copy.deepcopy(self.__feature_value_color_dict)

    def get_feature_binning(self,
                            feature_name):
        """
       Desc:
           Get's the feature's bin and labels for that given feature.

       Args:
           feature_name: str
               The given feature name of the feature

       Returns:
           Returns the bin and labels for the given feature; returns None if
           the feature name is not saved in the feature value color dict.
       """

        if feature_name in self.__feature_labels_bins_dict:
            return copy.deepcopy(self.__feature_labels_bins_dict[feature_name])
        else:
            return None

    def get_all_feature_binning(self):
        """
        Desc:
           Get's the entire dict of the feature's bins and labels.

        Returns:
            Returns a copy of the labels and bins for the given feature.
        """
        return copy.deepcopy(self.__feature_labels_bins_dict)

    def get_feature_value_representation(self):
        """
        Desc:
            Get's the entire dict of the feature's value colors.

        Returns:
            Returns a copy of feature's value colors.
        """
        return copy.deepcopy(self.__feature_value_representation)

    def get_dummy_encoded_features(self):
        """
        Desc:
            Get's all dummy encoded relationships.
        """
        return copy.deepcopy(self.__dummy_encoded_features)

    # --- Setters and Appending ---
    def set_target_feature(self,
                           feature_name):
        """
        Desc:
            Sets the value.

        Args:
            feature_name:
                Set the target feature.
        """
        self.__target_feature = copy.deepcopy(feature_name)


    def set_feature_to_bool(self,
                            feature_name):
        """
        Desc:
            Moves feature to bool set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """

        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__bool_features.add(name)

    def set_feature_to_integer(self,
                               feature_name):
        """
        Desc:
            Moves feature to integer set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__integer_features.add(name)

    def set_feature_to_float(self,
                             feature_name):
        """
        Desc:
            Moves feature to float set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__float_features.add(name)

    def set_feature_to_string(self,
                              feature_name):
        """
        Desc:
            Moves feature to string set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """

        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__string_features.add(name)

    def set_feature_to_categorical(self,
                                   feature_name):
        """
        Desc:
            Moves feature to categorical set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """

        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__categorical_features.add(name)

    def set_feature_to_datetime(self,
                                feature_name):
        """
        Desc:
            Moves feature to datetime set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """

        if isinstance(feature_name,str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__datetime_features.add(name)


    def set_feature_to_null_only_features(self,
                                          feature_name):
        """
        Desc:
            Moves feature to only null series data feature set.

        Args:
            feature_name:
                Feature name or a list of feature names to move to given set.
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.remove_feature(name)
            self.__null_only_features.add(name)

    def set_feature_binning(self,
                            feature_name,
                            bins,
                            labels):

        if not isinstance(labels,list):
            labels = list(labels)

        if not isinstance(bins,list):
            bins = list(bins)

        self.__feature_labels_bins_dict[feature_name] = dict()
        self.__feature_labels_bins_dict[feature_name]["bins"] = bins
        self.__feature_labels_bins_dict[feature_name]["labels"] = labels


    def add_new_bool_feature(self,
                             feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set bool

        Args:
            feature_name: str
                Name of the new feature
        """

        if isinstance(feature_name,str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__bool_features.add(name)

    def add_new_string_feature(self,
                               feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set string

        Args:
            feature_name: str
                Name of the new feature
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__string_features.add(name)

    def add_new_integer_feature(self,
                                feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set integer

        Args:
            feature_name: str
                Name of the new feature
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__integer_features.add(name)

    def add_new_float_feature(self,
                              feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set float

        Args:
            feature_name: str
                Name of the new feature
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__float_features.add(name)

    def add_new_categorical_feature(self,
                                    feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set categorical

        Args:
            feature_name: str
                Name of the new feature
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__categorical_features.add(name)

    def add_new_null_only_feature(self,
                                  feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set null only features

        Args:
            feature_name: str
                Name of the new feature
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__null_only_features.add(name)

    def add_new_datetime_feature(self,
                                  feature_name):
        """
        Desc:
            Adds a new feature/feature(s) to the feature set datetime

        Args:
            feature_name: str
                Name of the new feature
        """
        if isinstance(feature_name, str):
            feature_name = [feature_name]

        for name in feature_name:
            self.__datetime_features.add(name)

    def set_feature_colors(self,
                           feature_value_color_dict):
        """
        Desc:
            Passing in a dictionary of feature names to value to hex color to
            save to this object. Error checks the dict for proper values.

        Args:
            feature_value_color_dict:
                Dictionary of feature names to value to hex color.
                Ex: feature_value_color_dict["Sex"]["Male"] = "#ffffff"
        """

        # -----
        for feature_name, color_dict in feature_value_color_dict.items():

            if feature_name not in self.all_features():
                raise UnsatisfiedRequirments(f"The feature name '{feature_name}' "
                                             + "was not found in any of the past"
                                             " feature type sets!")

            if isinstance(feature_name,str):

                # -----
                if isinstance(color_dict,dict):

                    for feature_value, hex_color in color_dict.items():
                        if not isinstance(hex_color, str):
                            raise UnsatisfiedRequirments(f"The feature value must be a string; not a {type(hex_color)}")

                    self.__feature_value_color_dict[feature_name] = color_dict

                # -----
                # elif isinstance(color_dict, str):
                #     try:
                #         sns.color_palette(color_dict)
                #     except:
                #         raise ValueError(f"The value {color_dict} is not a proper seaborn color template!")
                #
                #     self.__feature_value_color_dict[feature_name] = color_dict

                # -----
                else:
                    raise UnsatisfiedRequirments("Expected to extract out a "
                                                 + f"dict from the feature '{feature_name}' "
                                                 + "values with accoiated color hex "
                                                 "values. Instead was found to"
                                                 + f" be a {type(color_dict)}.")
            else:
                raise UnsatisfiedRequirments(f"Expect the feature name to be a "
                                             + f"string instead was found to be {type(feature_name)}")


    def set_feature_value_representation(self,
                                         feature_value_representation):
        for feature_name in feature_value_representation:
            if feature_name not in self.__string_features:
                raise UnsatisfiedRequirments(f"Feature value assertions must be of type string.")

            if feature_name not in self.all_features():
                raise UnsatisfiedRequirments(f"'{feature_name}' doesn't exist in any features.")

        self.__feature_value_representation = copy.deepcopy(feature_value_representation)

    def set_feature_to_dummy_encoded(self,
                                     feature_name,
                                     dummy_encoded_list):
        self.__dummy_encoded_features[feature_name] = dummy_encoded_list

        for bool_feature in dummy_encoded_list:
            self.__bool_features.add(bool_feature)

    # --- Functions ---
    def feature_types_dict(self):
        feature_types = dict()

        # -----
        for feature_name in self.__string_features:
            feature_types[feature_name] = "string"

        for feature_name in self.__bool_features:
            feature_types[feature_name] = "bool"

        for feature_name in self.__integer_features:
            feature_types[feature_name] = "integer"

        for feature_name in self.__float_features:
            feature_types[feature_name] = "float"

        for feature_name in self.__datetime_features:
            feature_types[feature_name] = "datetime"

        for feature_name in self.__categorical_features:
            feature_types[feature_name] = "categorical"

        for feature_name in self.__null_only_features:
            feature_types[feature_name] = "null_only"

        return feature_types

    def feature_types_dataframe(self):
        features = list()
        feature_types = list()

        # -----
        for feature_name in self.__string_features:
            features.append(feature_name)
            feature_types.append("string")

        for feature_name in self.__bool_features:
            features.append(feature_name)
            feature_types.append("bool")

        for feature_name in self.__integer_features:
            features.append(feature_name)
            feature_types.append("integer")

        for feature_name in self.__float_features:
            features.append(feature_name)
            feature_types.append("float")

        for feature_name in self.__datetime_features:
            features.append(feature_name)
            feature_types.append("datetime")

        for feature_name in self.__categorical_features:
            features.append(feature_name)
            feature_types.append("category")

        for feature_name in self.__null_only_features:
            features.append(feature_name)
            feature_types.append("null only")

        dtypes_df = pd.DataFrame({'Data Types': feature_types})
        dtypes_df.index = features
        dtypes_df.index.name = "Features"

        return dtypes_df


    def remove_feature_from_dummy_encoded(self,
                                          feature_name):

        for bool_feature in self.__dummy_encoded_features[feature_name]:
            self.remove_feature(bool_feature)

        del self.__dummy_encoded_features[feature_name]


    def remove_feature(self,
                       feature_name):
        """
        Desc:
            Removes a feature from one of the feature sets.

        Args:
            feature_name:
                The given feature name to remove.
        """
        while True:
            try:
                self.__string_features.remove(feature_name)
                break
            except KeyError:
                pass

            try:
                self.__categorical_features.remove(feature_name)
                break
            except KeyError:
                pass

            try:
                self.__integer_features.remove(feature_name)
                break
            except KeyError:
                pass

            try:
                self.__float_features.remove(feature_name)
                break
            except KeyError:
                pass

            try:
                self.__datetime_features.remove(feature_name)
                break
            except KeyError:
                pass

            try:
                self.__bool_features.remove(feature_name)
                break
            except KeyError:
                pass

            try:
                self.__null_only_features.remove(feature_name)
                break
            except KeyError:
                pass

            raise KeyError(f"The feature {feature_name} doesn't exist inside any of DataFrameType's feature sets!!!")

    def display_features(self,
                         display_dataframes=False,
                         notebook_mode=False):
        """
        Desc:
            Display's the feature sets info.

        Args:
            display_dataframes:
                Creates a dataframe object to display feature information.

            notebook_mode:
                Determines if the dataframe can be displayed in a notebook
        """

        # Do a simple print of feature set info.
        if not display_dataframes:

            # -----
            if self.__string_features:
                print("String Features: {0}\n".format(
                    self.__string_features))

            if self.__categorical_features:
                print("Categorical Features: {0}\n".format(
                    self.__categorical_features))

            if self.__string_features or self.__categorical_features:
                print("---------"*10)

            # -----
            if self.__bool_features:
                print("Bool Features: {0}\n".format(
                    self.__bool_features))
                print("---------" * 10)

            if self.__datetime_features:
                print("Datetime Features: {0}\n".format(
                    self.__datetime_features))
                print("---------" * 10)

            # -----
            if self.__float_features | self.__integer_features:
                print("Numerical Features: {0}\n".format(
                    self.__float_features | self.__integer_features))
            if self.__integer_features:
                print("Integer Features: {0}\n".format(
                    self.__integer_features))

            if self.__float_features:
                print("Float Features: {0}\n".format(
                    self.__float_features))

            if self.__target_feature:
                print("---------" * 10)
                print("Target Feature: {0}\n".format(
                    self.__target_feature))

            if self.__null_only_features:
                print("---------" * 10)
                print("Null Only Feature: {0}\n".format(
                    self.__null_only_features))

        # Create dataframe object based on the feature sets.
        else:
            dtypes_df = self.feature_types_dataframe()

            if notebook_mode:
                display(dtypes_df)
            else:
                print(dtypes_df)

    def fix_numeric_features(self,
                             df,
                             notebook_mode=False,
                             display_results=False):
        """
        Desc:
            Attempts to move numerical features to the correct types by
            following the given priority que:
                1. Bool
                2. Categorical
                3. Float
                4. Int
                5. Do nothing
        Args:
            df: pd.Dataframe
                Pandas Dataframe object to update to correct types.

            notebook_mode: bool
                Boolean value to determine if any notebook functions can be used here.

            display_results: bool
                Display the table in priority order with flags.


        Note -
            This will not actually update the given dataframe. This object is
            a abstract representation of the dataframe.
        """

        features_flag_types = dict()
        for feature_name in df.columns:
            try:
                pd.to_numeric(df[feature_name])
            except ValueError:
                # Ignore all string features
                if feature_name in self.string_features():
                    continue

                # Features that must be these set types
                if feature_name in self.categorical_features():
                    continue

                if feature_name in self.bool_features():
                    continue

            feature_values = set(pd.to_numeric(df[feature_name],
                                 errors="coerce").dropna())

            if not len(feature_values):
                continue

            # Get flag's to push to priority que
            flag_dict = dict()
            flag_dict["Bool"] = self.__bool_check(feature_values)
            numeric_flag, float_flag, int_flag, category_flag = \
                self.__numeric_check(feature_values)
            flag_dict["Numeric"] = numeric_flag
            flag_dict["Float"] = float_flag
            flag_dict["Integer"] = int_flag
            flag_dict["Categorical"] = category_flag

            # Pass the flag dictionary to later be processed by the priority que.
            features_flag_types[feature_name] = flag_dict

        # Iterate on feature and changes based on priority que
        for feature_name, flag_dict in features_flag_types.items():

            # -----
            if flag_dict["Bool"]:
                self.set_feature_to_bool(feature_name)
                continue

            # -----
            elif flag_dict["Categorical"]:
                self.set_feature_to_categorical(feature_name)
                continue

            # -----
            elif flag_dict["Numeric"]:
                if flag_dict["Float"]:
                    self.set_feature_to_float(feature_name)
                    continue

                elif flag_dict["Integer"]:
                    self.set_feature_to_integer(feature_name)
                    continue

        if display_results:
            flag_df = pd.DataFrame.from_dict(features_flag_types,
                                             orient='index')
            if notebook_mode:
                display(flag_df)
            else:
                print(flag_df)

    def fix_string_features(self,
                            df,
                            notebook_mode=False):
        """
        Desc:
            Iterates through all string features and moves features to given types.
            May ask user question if it detects any conflicting string/numeric
            types.

        Args:
            df: pd.Dataframe
                Pandas dataframe object.

            notebook_mode: bool
                Will use the 'clear_output' notebook function if in notebook
                mode.
        """

        # Store types to convert
        type_conflict_dict = dict()

        # Currently this only performs
        for feature_name in self.string_features():

            # Float found
            float_flag = False

            # Keep track of all numeric features
            numeric_count = 0
            numeric_values = []

            # Keep track of all string features
            string_count = 0
            string_values = []

            # Keep track fof all datetime features
            datetime_count = 0
            datetime_values = []

            # Iterate through value counts
            for val, count in df[feature_name].dropna().value_counts().iteritems():

                numeric_check = False

                try:
                    float(val)
                    numeric_check = True
                except ValueError:
                    pass

                # Numeric check
                if isinstance(val, float) or isinstance(val,
                                                        int) or numeric_check == True:
                    numeric_values.append(val)
                    numeric_count += count

                    if isinstance(val, float):
                        float_flag = True

                    if numeric_check and isinstance(val, str):
                        if len(val.split(".")) == 2:
                            float_flag = True

                # String/Datetime check
                elif isinstance(val, str):

                    datetime_found = False
                    try:
                        parser.parse(val)
                        datetime_values.append(val)
                        datetime_count += count
                        datetime_found = True

                    except Exception as e:
                        pass

                    if not datetime_found:
                        string_values.append(val)
                        string_count += count

            # Must be a numeric type; find which type
            if numeric_count != 0 and string_count == 0 and datetime_count == 0:
                if float_flag:
                    type_conflict_dict[feature_name] = "float"
                else:
                    if self.__bool_check(numeric_values):
                        type_conflict_dict[feature_name] = "bool"

                    elif self.__categorical_check(numeric_values):
                        type_conflict_dict[feature_name] = "category"

                    else:
                        type_conflict_dict[feature_name] = "integer"

            # Must be a string type
            elif numeric_count == 0 and string_count != 0 and datetime_count == 0:

                if self.__bool_string_values_check(string_values):
                    type_conflict_dict[feature_name] = "bool"
                else:
                    type_conflict_dict[feature_name] = "string"

            # Must be a datetime
            elif numeric_count == 0 and string_count == 0 and datetime_count != 0:
                type_conflict_dict[feature_name] = "datetime"

            # A conflict is found; have the user work it out.
            else:
                print("Type conflict found!")
                print(f"Feature Name: '{feature_name}'")
                print("---" * 10)

                # -----
                print("Numeric Value Info")
                print(f"\tNumeric count: {numeric_count}")
                print(f"\tNumeric percentage: {(numeric_count / (numeric_count + string_count + datetime_count)) * 100:.3f}%")
                print(f"\tNumeric values: {numeric_values}\n")

                # -----
                print("String Value Info")
                print(f"\tString count: {string_count}")
                print(f"\tString percentage: {(string_count / (numeric_count + string_count + datetime_count)) * 100:.3f}%")
                print(f"\tString values: {string_values}\n")

                # -----
                print("Datetime Value Info")
                print(f"\tString count: {datetime_count}")
                print(f"\tString percentage: {(datetime_count / (numeric_count + string_count + datetime_count)) * 100:.3f}%")
                print(f"\tString values: {datetime_values}\n")

                # Get user input for handling
                print(
                    "You can use the first character of the option for input.\n")
                user_input = input(
                    "\nMove feature to numeric or string and replace any "
                    "conflicts with nulls.\n* Numeric\n* String\n* Datetime\n"
                    "* Ignore\nInput: ")
                user_input = user_input.lower()


                # Clear last user output. If in notebook mode. (A clean notebook is a happy notebook :).)
                if notebook_mode:
                    clear_output()

                # -----

                if not len(user_input):
                    print(f"Ignoring feature '{feature_name}")

                # -----
                elif user_input[0] == "s":
                    type_conflict_dict[feature_name] = "string"

                # -----
                elif user_input[0] == "n":
                    if float_flag:
                        type_conflict_dict[feature_name] = "float"
                    else:
                        numeric_values = set(pd.to_numeric(numeric_values,
                                                       errors="coerce").dropna())
                        if self.__bool_check(numeric_values):
                            type_conflict_dict[feature_name] = "bool"

                        elif self.__categorical_check(numeric_values):
                            type_conflict_dict[feature_name] = "category"

                        else:
                            type_conflict_dict[feature_name] = "integer"

                # -----
                elif user_input[0] == "d":
                    type_conflict_dict[feature_name] = "datetime"

                # -----
                else:
                    print(f"Ignoring feature '{feature_name}")

        # Iterate on all features
        for feature_name, feature_type in type_conflict_dict.items():

            moved_feature = False

            if feature_type == "string":

                if feature_name not in self.__string_features:
                    self.set_feature_to_string(feature_name)
                    moved_feature = True

            elif feature_type == "datetime":

                if feature_name not in self.__datetime_features:
                    self.set_feature_to_datetime(feature_name)
                    moved_feature = True

            elif feature_type == "integer":

                if feature_name not in self.__integer_features:
                    self.set_feature_to_integer(feature_name)
                    moved_feature = True

            elif feature_type == "category":

                if feature_name not in self.__categorical_features:
                    self.set_feature_to_categorical(feature_name)
                    moved_feature = True

            elif feature_type == "bool":

                if feature_name not in self.__bool_features:
                    self.set_feature_to_bool(feature_name)
                    moved_feature = True

            elif feature_type == "float":

                if feature_name not in self.__float_features:
                    self.set_feature_to_float(feature_name)
                    moved_feature = True

            else:
                raise TypeError("An unknown type was passed!")

            if moved_feature:
                print(f"\nMoving feature '{feature_name}' to type {feature_type}.")

    def fix_nan_features(self,
                         df):
        """
        Desc:
            Attempts to get the data type of the feature if there were no nans
            inside it.

        Args:
            df: pd.DataFrame
                Pandas Dataframe object.
        """
        nan_features = [feature for feature, nan_found in
                        df.isna().any().items() if nan_found]

        # The features that are found to be floats should partially merge with
        # features with nulls.
        float_features = set(
            df.select_dtypes(include=["float"]).columns)

        for feature_name in nan_features:

            # Ignore string features
            if feature_name in self.__string_features:
                continue

            feature_values = list(set(df[feature_name].dropna().sort_values(
                ascending=True)))

            # Convert to bool if possible
            if self.__bool_check(feature_values):
                self.set_feature_to_bool(feature_name)

            # Convert numeric to proper types (int,float,categorical)
            elif feature_name in float_features:

                numeric_flag, float_flag, int_flag, category_flag = self.__numeric_check(feature_values)

                if numeric_flag:
                    if category_flag:
                        self.set_feature_to_categorical(feature_name)

                    elif int_flag:
                        self.set_feature_to_integer(feature_name)

                    elif float_flag:
                        self.set_feature_to_float(feature_name)

    def create_json_file_representation(self,
                                        directory_path,
                                        filename):
        """
        Desc:
            Creates a json representation of the current objects feature sets
            and the targeted feature.

        Args:
            directory_path: string
                Absolute directory path.

            filename: string
                File's given name
        """
        type_features = dict()
        for feature_name in self.bool_features():
            if "bool" in type_features.keys():
                type_features["bool"].append(feature_name)
            else:
                type_features["bool"] = [feature_name]

        for feature_name in self.string_features():
            if "string" in type_features.keys():
                type_features["string"].append(feature_name)
            else:
                type_features["string"] = [feature_name]

        for feature_name in self.integer_features():
            if "integer" in type_features.keys():
                type_features["integer"].append(feature_name)
            else:
                type_features["integer"] = [feature_name]

        for feature_name in self.float_features():
            if "float" in type_features.keys():
                type_features["float"].append(feature_name)
            else:
                type_features["float"] = [feature_name]

        for feature_name in self.categorical_features():
            if "categorical" in type_features.keys():
                type_features["categorical"].append(feature_name)
            else:
                type_features["categorical"] = [feature_name]

        for feature_name in self.datetime_features():
            if "datetime" in type_features.keys():
                type_features["datetime"].append(feature_name)
            else:
                type_features["datetime"] = [feature_name]

        type_features["target"] = self.__target_feature

        type_features["feature_labels_bins"] = self.__feature_labels_bins_dict

        type_features["feature_values_colors"] = self.__feature_value_color_dict

        type_features["feature_value_representation"] = self.__feature_value_representation

        type_features["label_encoder"] = self.__label_encoder
        type_features["label_decoder"] = self.__label_decoder

        dict_to_json_file(type_features,
                          directory_path,
                          filename)

    def init_on_json_file(self,
                          filepath):
        """
        Desc:
            Initialize object based on a json file.

        Args:
            filepath:
                Absolute path to the given file.

        Note:
            Structure must be the given type to the list of associated features.
        """
        type_features = json_file_to_dict(filepath)

        # Reset all sets and target
        self.__bool_features = set()
        self.__string_features = set()
        self.__categorical_features = set()
        self.__integer_features = set()
        self.__float_features = set()
        self.__datetime_features = set()

        self.__target_feature = None

        # -----
        self.__feature_value_color_dict = dict()

        # -----
        self.__feature_value_representation = dict()
        self.__label_encoder = dict()
        self.__label_decoder = dict()

        tmp_feature_value_color_dict = dict()

        # Iterate through dict given types to feature lists
        for type, obj  in type_features.items():

            if type == "bool":
                self.__bool_features = set(obj)

            # -----
            elif type == "integer":
                self.__integer_features = set(obj)

            # -----
            elif type == "float":
                self.__float_features = set(obj)

            # -----
            elif type == "string":
                self.__string_features = set(obj)

            # -----
            elif type == "categorical":
                self.__categorical_features = set(obj)

            # -----
            elif type == "datetime":
                self.__datetime_features = set(obj)

            # Extract target
            elif type == "target":
                self.__target_feature = obj

            # Extract color dict.(Naming convention get's a little screwed up here.)
            elif type == "feature_values_colors":
                tmp_feature_value_color_dict = obj

            # Extract out the feature value representation
            elif type == "feature_value_representation":
                self.__feature_value_representation = obj

            elif type == "label_decoder":
                self.__label_decoder = obj

            elif type == "label_encoder":
                self.__label_encoder = obj

            elif type == "feature_labels_bins":
                self.__feature_labels_bins_dict = obj

            else:
                raise ValueError(f"Unknown type {type} was found!")

        # Convert any values that are supposed to numeric back to numeric in colors
        for feature_name,value_color_dict in tmp_feature_value_color_dict.items():
            if feature_name not in self.all_features():
                raise ValueError(f"Unknown feature '{feature_name}' found in color dict for features!")
            else:
                tmp_value_color_dict = copy.deepcopy(value_color_dict)
                for feature_val,color in value_color_dict.items():
                    try:
                        tmp_value_color_dict[int(feature_val)] = color
                        del tmp_value_color_dict[feature_val]
                    except ValueError:
                        pass
                self.__feature_value_color_dict[feature_name] = tmp_value_color_dict

        # Converting to numeric on decoder's categories
        tmp_decoder = self.__label_decoder
        for feature_name,cat_val_dict in tmp_decoder.items():

            cat_val_dict = copy.deepcopy(cat_val_dict)
            for cat, val in cat_val_dict.items():
                self.__label_decoder[feature_name][int(cat)] = self.__label_decoder[feature_name][cat]
                del self.__label_decoder[feature_name][cat]

    def set_encoder_for_features(self,
                                 df,
                                 categorical_value_dict=dict()):
        """
        Desc:
            Create's encoder dict for strings and categories.

        Args:
            df: pd.DataFrame
                Pandas dataframe.

            categorical_value_dict: dict
                Relationship between category and string label.

        Note:
            Can handle a feature that have categories and strings in same series.
        """

        if len(set(df.columns) ^ set(self.all_features())) > 0:
            raise UnsatisfiedRequirments("The given Dataframe's features should "
                                         "be the same as the features saved in "
                                         "the DataFrameTypes.")
        # Reset dict
        self.__label_encoder = dict()

        # Get all string and categorical features
        categorical_string_features = self.categorical_features() | self.string_features()

        for feature_name in categorical_string_features:

            self.__label_encoder[feature_name] = dict()

            # Order feature values
            feature_values = [str(val) for val in
                              df[feature_name].dropna().value_counts(
                                  sort=False).index.to_list()]
            feature_values.sort()

            # Convert to category (int) if possible
            for i in range(0, len(feature_values)):
                try:
                    feature_values[i] = int(float(feature_values[i]))
                except ValueError:
                    pass

            # Pre-defined category values
            default_categories = [i for i in range(0,
                                                   len(feature_values))]

            # Check if user defined any category to string relationships
            if feature_name in categorical_value_dict.keys():
                for cat, label in categorical_value_dict[feature_name].items():

                    # Error checks
                    if label == "":
                        raise UnsatisfiedRequirments(
                            f"Can't replace category value with empty space! Found on feature {feature_name}")

                    if isinstance(label, int):
                        raise UnsatisfiedRequirments(
                            f"Can't replace category value with a number based value! Found on feature {feature_name}")

                    if not isinstance(cat, int):
                        raise UnsatisfiedRequirments(
                            f"Category value must be a number based value! Found on feature {feature_name}")

                    if feature_name not in self.__categorical_features:
                        raise UnsatisfiedRequirments(
                            f"User defined a feature name that doesn't appear to be a '{feature_name}'")

                    self.__label_encoder[feature_name][label] = int(cat)

                    # Remove category and label from collection objects
                    if cat in default_categories:
                        default_categories.remove(cat)

                    if cat in feature_values:
                        feature_values.remove(cat)

                    if label in feature_values:
                        feature_values.remove(label)

            # Encode strings and categories
            cat_count = 0
            for val in feature_values:

                if isinstance(val, int):
                    self.__label_encoder[feature_name][val] = val
                else:
                    self.__label_encoder[feature_name][val] = default_categories[
                        cat_count]
                    cat_count += 1

        # Inverse dict
        self.__label_decoder = dict()
        for feature_name, label_val_dict in self.__label_encoder.items():
            self.__label_decoder[feature_name] = dict()
            for label, cat in label_val_dict.items():
                self.__label_decoder[feature_name][cat] = label

    def __bool_check(self,
                     feature_values):
        """
        Desc:
            Determines if the feature type of this column is a boolean based on
            the numeric values passed to it.

        Args:
            feature_values: collection
                Distinct values of the given feature.

        Returns:
            Returns true or false if the values are boolean.
        """
        # Convert to bool if possible
        if len(feature_values) == 1 and (
                0.0 in feature_values or 1.0 in feature_values):
            return True

        # Second bool check
        elif len(feature_values) == 2 and (
                0.0 in feature_values and 1.0 in feature_values):
            return True
        else:
            return False

    def __bool_string_values_check(self,
                                   feature_values):
        """
        Desc:
            Checks if a collection of strings can be considered a bool feature
            based on the amount of strings and the values of those strings.

        Args:
            feature_values: collection
                Collection object of strings.

        Returns:
            Returns true or false if the values can be considered a bool.
        """

        if len(feature_values) > 2:
            return False

        found_true_value = False
        found_false_value = False

        for val in feature_values:

            if not isinstance(val,str):
                continue

            val = val.lower()

            # Determine if val is true
            if not found_true_value:

                # Check if the string already exist in the defined set
                if val in BOOL_STRINGS.TRUE_STRINGS:
                    found_true_value = True
                    continue
                else:
                    # Attempt to find synonyms of the defined words to compare to
                    # the iterable string
                    for true_string in BOOL_STRINGS.TRUE_STRINGS:

                        if len(true_string) < 2:
                            continue

                        for syn in get_synonyms(true_string):
                            if syn == val:
                                found_true_value = True
                                continue

            # -----
            if not found_false_value:

                # -----
                if val in BOOL_STRINGS.FALSE_STRINGS:
                    found_false_value = True
                    continue
                else:
                    # -----
                    for false_string in BOOL_STRINGS.FALSE_STRINGS:

                        if len(false_string) < 2:
                            continue

                        for syn in get_synonyms(false_string):
                            if syn == val:
                                found_false_value = True
                                continue

        if len(feature_values) == 2:
            return found_true_value and found_false_value
        else:
            return found_true_value or found_false_value

    def __categorical_check(self,
                            feature_values):
        """
        Desc:
            Check to see if the feature's value can be consider a category.

        Args:
            feature_values:
                Distinct values of the given feature.

        Returns:
            Returns true or false if the values can be categorical.
        """

        feature_values = copy.deepcopy(set(feature_values))

        min_val = min(feature_values)
        max_val = max(feature_values)

        if sum(feature_values) != ((min_val + max_val) / 2) * len(
                set(feature_values)):
            return False
        else:
            return True

    def __numeric_check(self,
                        feature_values):
        """
        Desc:
            Checks if the features can be a numerical value and other numerical
            types like floats, ints, and categories.

        Args:
            feature_values:
                Distinct values of the given feature.

        Returns:
            Boolean values return as such:
                numerical_check, float_check, int_check, categorical_check
        """

        if not len(feature_values):
            return False, False, False, False


        # -----
        float_check = False
        categorical_total_sum = 0
        categorical_check = True

        feature_values = copy.deepcopy(set(feature_values))

        # Iterate through all values
        for val in feature_values:

            # Ignore None values
            if not val:
                continue
            try:
                float(val)
            except ValueError:
                return False, False, False, False

            if categorical_check:
                # Check if float
                str_val = str(val)
                tokens = str_val.split(".")
                if len(tokens) > 1 and int(tokens[1]) > 0:
                    float_check = True
                    categorical_check = False
                else:
                    categorical_total_sum += val
                    continue

        # Check if categorical
        if categorical_check:

            min_val = min(feature_values)
            max_val = max(feature_values)

            if categorical_total_sum != ((min_val + max_val)/2) * len(feature_values):
                categorical_check = False

        return True, float_check, not float_check, categorical_check


    def __eq__(self,
               other):

        if isinstance(other,DataFrameTypes):

            if self.feature_types_dict() != other.feature_types_dict():
                return False

            elif self.get_label_encoder() != other.get_label_encoder():
                return False

            elif self.get_label_decoder() != other.get_label_decoder():
                return False

            elif self.get_all_feature_colors() != other.get_all_feature_colors():
                return False

            elif self.get_all_feature_binning() != other.get_all_feature_binning():
                return False

            elif self.target_feature() != other.target_feature():
                return False

            else:
                return True

        else:
            return False