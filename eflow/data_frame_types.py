import copy
from IPython.display import display, HTML
from dateutil import parser

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

        # Data type assertions without nulls
        if ignore_nulls and df.isnull().values.any():
            tmp_df = copy.deepcopy(df)
            nan_columns = [feature for feature, nan_found in
                           tmp_df.isna().any().items() if nan_found]
            self.__make_type_assertions_after_ignore_nan(tmp_df,
                                                         nan_columns)
        else:
            tmp_df = df

        self.__all_columns = df.columns.tolist()
        # Grab features based on there types
        self.__bool_features = set(tmp_df.select_dtypes(include=["bool"]).columns)
        self.__string_features = set(
            tmp_df.select_dtypes(include=["object"]).columns)
        self.__categorical_features = set(df.select_dtypes(include=["category"]).columns)
        self.__integer_features = set(tmp_df.select_dtypes(include=["int"]).columns)
        self.__float_features = set(tmp_df.select_dtypes(include=["float"]).columns)
        self.__numerical_features = self.__float_features | self.__integer_features
        self.__datetime_features = set(
            tmp_df.select_dtypes(include=["datetime"]).columns)

        # Extra functionality
        self.__target_feature = None

        # Attempt to init target column
        if target_feature:
            if target_feature in tmp_df.columns:
                self.__target_feature = target_feature
            else:
                print("WARNING!!!: THE FEATURE {0} "
                      "DOES NOT EXIST IN THIS DATASET!".format(target_feature))

        # # Create one hot encoded features
        # if self.__string_features and self.__categorical_features:
        #     for col_feature in self.__string_features:
        #
        #         if col_feature is not self.__target_feature:
        #             self.__one_hot_encoded_names[col_feature] = list()
        #             for value_of_col in set(tmp_df[col_feature].values):
        #                 if isinstance(value_of_col,str):
        #                     self.__one_hot_encoded_names[col_feature].append(
        #                         col_feature + "_" + value_of_col.replace(" ",
        #                                                                  "_"))
        if display_init:
            self.display_all()
        features_not_captured = set(tmp_df.columns)
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
        if exclude_target:
            return [col_feature for col_feature in self.__numerical_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__numerical_features)

    def get_integer_features(self,
                             exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__integer_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__integer_features)

    def get_float_features(self,
                           exclude_target=False):
        if exclude_target:
             return [col_feature for col_feature in self.__float_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__float_features)

    def get_categorical_features(self,
                                 exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__categorical_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__categorical_features)

    def get_string_features(self,
                            exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__string_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__string_features)

    def get_bool_features(self,
                          exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__bool_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__bool_features)

    def get_datetime_features(self,
                              exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__datetime_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__datetime_features)

    def get_all_features(self):
        return copy.deepcopy(self.__all_columns)

    def get_target(self):
        return copy.deepcopy(self.__target_feature)

    # --- Remover
    def remove(self,
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
            print("Categorical Features: {0}\n".format(
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
            feature_values = list(set(df[feature_name].dropna()))

            # Convert to bool if possible
            if len(feature_values) == 1 and (
                    0.0 in feature_values or 1.0 in feature_values):
                df[feature_name] = df[feature_name].astype(bool)

            # Second bool check
            elif len(feature_values) == 2 and (
                    0.0 in feature_values and 1.0 in feature_values):
                df[feature_name] = df[feature_name].astype(bool)

            # Convert numeric to proper types (int,float,categorical)
            elif feature_name in float_features:
                feature_values = [str(i) for i in feature_values]
                convert_to_float = False
                for str_val in feature_values:
                    tokens = str_val.split(".")

                    if len(tokens) > 1 and len(tokens[1]) > 1 or \
                            int(tokens[1]) > 0:
                        convert_to_float = True
                if convert_to_float:
                    df[feature_name].fillna(0, inplace=True)
                    df[feature_name] = df[feature_name].astype(float)
                else:
                    df[feature_name].fillna(0, inplace=True)
                    df[feature_name] = df[feature_name].dropna().astype(int)

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
