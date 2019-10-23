import copy
from IPython.display import display, HTML

import numpy as np
import pandas as pd

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class DataFrameTypes:

    """
        Seperates the features based off of dtypes to better keep track of
        feature types and helps make type assertions.
    """

    def __init__(self,
                 df,
                 target_feature=None,
                 ignore_nulls=False):
        """
        Args:
            df:
                Pandas dataframe object.

            target_feature:
                If the project is using a supervised learning approach we can
                specify the target column. (Note: Not required)

            ignore_nulls:
                If set to true than a temporary dataframe is created with each
                feature removes it's nan values to assert what data type the series
                object would be without nans existing inside it.
        """
        self.__all_columns = df.columns.tolist()
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

        # Extra functionality
        self.__target_feature = None

        # Data type assertions without nulls
        if ignore_nulls and df.isnull().values.any():
            nan_features = [feature for feature, nan_found in
                            df.isna().any().items() if nan_found]
            self.__make_type_assertions_after_ignore_nan(df,
                                                         nan_features)

        # Attempt to init target column
        if target_feature:
            if target_feature in df.columns:
                self.__target_feature = target_feature
            else:
                raise KeyError(f"The given target feature: \'{target_feature}\' does not exist!")

        # Error
        features_not_captured = set(df.columns)
        for col_feature in ((self.__float_features | self.__integer_features) |
                            self.__string_features |
                            self.__bool_features |
                            self.__datetime_features |
                            self.__categorical_features):
            features_not_captured.remove(col_feature)

        if features_not_captured:
            print("ERROR UNKNOWN FEATURE(S) TYPE(S) FOUND!\n{0}".format(
                features_not_captured))

    # --- Getters ---
    def get_numerical_features(self,
                               exclude_target=False):
        """
        Desc:
            Gets all numerical features chosen by the object.

        Args:
            exclude_target:
                If the target feature is an numerical (int/float); then it will be ignored
                when passing back the set.

        Returns:
            Returns a set of all numerical features chosen by the object.
        """
        if exclude_target:
            tmp_set = self.__float_features | self.__integer_features

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return self.__float_features | self.__integer_features

    def get_integer_features(self,
                             exclude_target=False):
        """
        Desc:
            All integer features chosen by df_features.

        Args:
            exclude_target:
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

    def get_float_features(self,
                           exclude_target=False):
        """
        Desc:
            All float features chosen by df_features.

        Args:
            exclude_target:
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

    def get_categorical_features(self,
                                 exclude_target=False):
        """
        Desc:
            All categorical features chosen by df_features.

        Args:
            exclude_target:
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

    def get_string_features(self,
                            exclude_target=False):
        """
        Desc:
            All string features chosen by df_features.

        Args:
            exclude_target:
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

    def get_bool_features(self,
                          exclude_target=False):

        """
        Desc:
            All bool features chosen by df_features.

        Args:
            exclude_target:
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

    def get_datetime_features(self,
                              exclude_target=False):
        """
        Desc:
            All datetime features chosen by df_features.

        Args:
            exclude_target:
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

    def get_all_features(self):
        """
        Returns:
            Returns all found features.
        """
        return copy.deepcopy(self.__all_columns)

    def get_target_feature(self):
        """
        Returns:
            Gets the target feature.
        """
        return copy.deepcopy(self.__target_feature)

    # --- Setters ---
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
                Feature name to move to given set.
        """
        self.remove_feature(feature_name)
        self.__bool_features.add(feature_name)

    def set_feature_to_integer(self,
                               feature_name):
        """
        Desc:
            Moves feature to integer set.

        Args:
            feature_name:
                Feature name to move to given set.
        """
        self.remove_feature(feature_name)
        self.__integer_features.add(feature_name)

    def set_feature_to_float(self,
                             feature_name):
        """
        Desc:
            Moves feature to float set.

        Args:
            feature_name:
                Feature name to move to given set.
        """
        self.remove_feature(feature_name)
        self.__float_features.add(feature_name)

    def set_feature_to_string(self,
                              feature_name):
        """
        Desc:
            Moves feature to string set.

        Args:
            feature_name:
                Feature name to move to given set.
        """
        self.remove_feature(feature_name)
        self.__string_features.add(feature_name)

    def set_feature_to_categorical(self,
                                   feature_name):
        """
        Desc:
            Moves feature to categorical set.

        Args:
            feature_name:
                Feature name to move to given set.
        """
        self.remove_feature(feature_name)
        self.__categorical_features.add(feature_name)

    def set_feature_to_datetime(self,
                                feature_name):
        """
        Desc:
            Moves feature to datetime set.

        Args:
            feature_name:
                Feature name to move to given set.
        """
        self.remove_feature(feature_name)
        self.__datetime_features.add(feature_name)

    # --- Functions ---
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
                self.__string_features.remove(feature_name)
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

            raise KeyError("This feature doesn't exist inside any of DataFrameType's feature sets!!!")

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

            print("---------" * 10)
            if self.__target_feature:
                print("Target Feature: {0}\n".format(
                    self.__target_feature))

        # Create dataframe object based on the feature sets.
        else:
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

            dtypes_df = pd.DataFrame({'Data Types': feature_types})
            dtypes_df.index = features
            dtypes_df.index.name = "Features"

            if notebook_mode:
                display(dtypes_df)
            else:
                print(dtypes_df)


    def __make_type_assertions_after_ignore_nan(self,
                                                df,
                                                nan_features):
        """
        Desc:
            Attempts to get the data type of the feature if there were no nans
            inside it.

        Args:
            df:
                Pandas Dataframe object.

            nan_features:
                Features that contain nulls.
        """
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
            if len(feature_values) == 1 and (
                    0.0 in feature_values or 1.0 in feature_values):
                self.set_feature_to_bool(feature_name)

            # Second bool check
            elif len(feature_values) == 2 and (
                    0.0 in feature_values and 1.0 in feature_values):
                self.set_feature_to_bool(feature_name)

            # Convert numeric to proper types (int,float,categorical)
            elif feature_name in float_features:
                feature_values = [str(i) for i in feature_values]

                # Check if feature would be interrupted as a float feature.
                convert_to_float = False
                for str_val in feature_values:
                    tokens = str_val.split(".")

                    if len(tokens) > 1 and int(tokens[1]) > 0:
                        convert_to_float = True
                        break

                if convert_to_float:
                    self.set_feature_to_float(feature_name)
                else:
                    # Check if feature would be interrupted as a categorical feature.
                    last_val = None
                    convert_to_categorical = True
                    for val in feature_values:
                        if not last_val:
                            last_val = val
                            continue
                        if np.abs(int(val.split(".")[0]) - int(last_val.split(".")[0])) != 1:
                            convert_to_categorical = False
                            break
                        else:
                            last_val = val

                    if convert_to_categorical:
                        self.set_feature_to_categorical(feature_name)
                    else:
                        self.set_feature_to_integer(feature_name)