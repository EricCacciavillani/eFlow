import copy
from IPython.display import display, HTML
from dateutil import parser
import numpy as np

class DataFrameTypes:

    """
        Seperates the features based off of dtypes
        to better keep track of feature types.
    """

    def __init__(self,
                 df,
                 target_feature=None,
                 ignore_nulls=False,
                 display_init=True):
        """

        df:
            Pandas dataframe object.

        target_feature:
            If the project is using a supervised learning approach we can
            specify the target column. (Note: Not required)

        ignore_nulls:
            If set to true than a temporary dataframe is created with each
            feature removes it's nan values to assert what data type the series
            object would be without nans existing inside it.

        display_init:
            Display results when object init
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
        self.__numerical_features = self.__float_features | self.__integer_features
        self.__datetime_features = set(
            df.select_dtypes(include=["datetime"]).columns)

        # Extra functionality
        self.__target_feature = None

        # Data type assertions without nulls
        if ignore_nulls and df.isnull().values.any():
            nan_columns = [feature for feature, nan_found in
                           df.isna().any().items() if nan_found]
            self.__make_type_assertions_after_ignore_nan(df,
                                                         nan_columns)

        # Attempt to init target column
        if target_feature:
            if target_feature in df.columns:
                self.__target_feature = target_feature
            else:
                raise KeyError(f"The given target feature: \'{target_feature}\' does not exist!")

        # -------
        if display_init:
            self.display_all()

        # Error
        features_not_captured = set(df.columns)
        for col_feature in (self.__numerical_features |
                            self.__string_features |
                            self.__bool_features |
                            self.__datetime_features |
                            self.__categorical_features):
            features_not_captured.remove(col_feature)

        if features_not_captured:
            print("ERROR UNKNOWN FEATURE(S) TYPE(S) FOUND!\n{0}".format(
                features_not_captured))

    # --- Getters
    def get_numerical_features(self,
                               exclude_target=False):
        """
        exclude_target:
            If the target feature is an numerical (int/float); then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all numerical features
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__numerical_features)

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)
            return tmp_set
        else:
            return copy.deepcopy(self.__numerical_features)

    def get_integer_features(self,
                             exclude_target=False):
        """
        exclude_target:
            If the target feature is an integer; then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all integer features.
        """
        print(exclude_target)
        if exclude_target:
            tmp_set = copy.deepcopy(self.__integer_features)

            # Target feature never init
            if self.__target_feature:
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
        exclude_target:
            If the target feature is an float; then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all float features.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__float_features)

            # Target feature never init
            if self.__target_feature:
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
        exclude_target:
            If the target feature is an categorical; then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all categorical features.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__categorical_features)

            # Target feature never init
            if self.__target_feature:
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
        exclude_target:
            If the target feature is an string; then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all string features.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__string_features)

            # Target feature never init
            if self.__target_feature:
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
        exclude_target:
            If the target feature is an bool; then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all bool features.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__bool_features)

            # Target feature never init
            if self.__target_feature:
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
        exclude_target:
            If the target feature is an datetime; then it will be ignored
            when passing back the set.

        Returns/Desc:
            Returns a set of all datetime features.
        """
        if exclude_target:
            tmp_set = copy.deepcopy(self.__datetime_features)

            # Target feature never init
            if self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__datetime_features)

    def get_all_features(self):
        """
        Returns/Desc:
            Returns all found features
        """
        return copy.deepcopy(self.__all_columns)

    def get_target_feature(self):
        """
        Returns/Desc:
            Get target feature
        """
        return copy.deepcopy(self.__target_feature)

    def set_target_feature(self,
                           feature_name):
        """
        feature_name:
            Set the target feature.

        Returns/Desc:
            Sets the value
        """
        self.__target_feature = copy.deepcopy(feature_name)

    # ---
    def remove_feature(self,
                       feature_name):
        try:
            self.__string_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__string_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__numerical_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__integer_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__float_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__datetime_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__bool_features.remove(feature_name)
        except KeyError:
            pass

    def display_all(self):
        """
            Display all sets
        """

        # Display category based features
        if self.__string_features:
            print("String Features: {0}\n".format(
                self.__string_features))

        if self.__categorical_features:
            print("Categorical Features: {0}\n".format(
                self.__categorical_features))

        if self.__string_features or self.__categorical_features:
            print("---------"*10)

        if self.__bool_features:
            print("Bool Features: {0}\n".format(
                self.__bool_features))
            print("---------" * 10)

        if self.__datetime_features:
            print("Datetime Features: {0}\n".format(
                self.__datetime_features))
            print("---------" * 10)

        # ---
        if self.__numerical_features:
            print("Numerical Features: {0}\n".format(
                self.__numerical_features))
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

    def __make_type_assertions_after_ignore_nan(self,
                                                df,
                                                nan_columns):
        float_features = set(
            df.select_dtypes(include=["float"]).columns)

        for feature_name in nan_columns:
            feature_values = list(set(df[feature_name].dropna().sort_values(
                ascending=True)))

            # Convert to bool if possible
            if len(feature_values) == 1 and (
                    0.0 in feature_values or 1.0 in feature_values):
                self.remove_feature(feature_name)
                self.__bool_features.add(feature_name)

            # Second bool check
            elif len(feature_values) == 2 and (
                    0.0 in feature_values and 1.0 in feature_values):
                self.remove_feature(feature_name)
                self.__bool_features.add(feature_name)

            # Convert numeric to proper types (int,float,categorical)
            elif feature_name in float_features:
                feature_values = [str(i) for i in feature_values]
                convert_to_float = False
                for str_val in feature_values:
                    tokens = str_val.split(".")

                    if len(tokens) > 1 and int(tokens[1]) > 0:
                        convert_to_float = True
                        break
                if convert_to_float:
                    self.remove_feature(feature_name)
                    self.__numerical_features.add(feature_name)
                    self.__float_features.add(feature_name)
                else:
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
                        self.remove_feature(feature_name)
                        self.__categorical_features.add(feature_name)
                    else:
                        self.remove_feature(feature_name)
                        self.__numerical_features.add(feature_name)
                        self.__integer_features.add(feature_name)

            # Attempt to convert string to datetime
            else:
                try:
                    _ = [parser.parse(val) for val in feature_values]
                    df[feature_name].fillna(feature_values[0],
                                            inplace=True)
                    df[feature_name] = [parser.parse(val)
                                        for val in df[feature_name]]
                except ValueError:
                    pass
