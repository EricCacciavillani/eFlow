from eflow.utils.sys_utils import create_json_file_from_dict,json_file_to_dict

import copy
from IPython.display import display, HTML

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
        Seperates the features based off of dtypes to better keep track of
        feature types and helps make type assertions.
    """

    def __init__(self,
                 df,
                 target_feature=None,
                 ignore_nulls=False,
                 fix_numeric_features=False,
                 fix_string_features=False,
                 notebook_mode=False):
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

            fix_numeric_features:
                Will attempt to convert all numeric features to the most proper
                numerical types.

            fix_string_features:
                Will attempt to convert all string features to ALL proper types.

            notebook_mode:
                Determine if any notebook functions can be used here.
        """
        if df is None:
            df = pd.DataFrame({})

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

        # Target feature for machine learning projects
        self.__target_feature = None

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
        for col_feature in ((self.__float_features | self.__integer_features) |
                            self.__string_features |
                            self.__bool_features |
                            self.__datetime_features |
                            self.__categorical_features):
            features_not_captured.remove(col_feature)

        if features_not_captured:
            print("ERROR UNKNOWN FEATURE(S) TYPE(S) FOUND!\n{0}".format(
                features_not_captured))

        if fix_string_features:
            self.fix_string_features(df,
                                     notebook_mode)

        if fix_numeric_features:
            self.fix_numeric_features(df)



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

    def get_all_features(self,
                         exclude_target=False):
        """
        Desc:
            Returns all features found in the dataset.

        Args:
            exclude_target:
                If the target feature is an datetime; then it will be ignored
                when passing back the set.
        Returns:
            Returns all found features.
        """

        if exclude_target:
            tmp_set = copy.deepcopy(self.__all_columns)

            # Target feature never init
            if not self.__target_feature:
                raise KeyError("Target feature was never initialized")

            # Check if target exist in set
            if self.__target_feature in tmp_set:
                tmp_set.remove(self.__target_feature)

            return tmp_set
        else:
            return copy.deepcopy(self.__all_columns)


    def get_target_feature(self):
        """
        Returns:
            Gets the target feature.
        """
        return copy.deepcopy(self.__target_feature)

    @property
    def target_feature(self):
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

    def fix_numeric_features(self,
                             df):
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
            df:
                Pandas Dataframe object to update to correct types.

        Note:
            This will not actually update the given dataframe. This object is
            a abstract representation of the dataframe.
        """

        features_flag_types = dict()
        for feature_name in df.columns:

            # Ignore all string features
            if feature_name in self.get_string_features():
                continue

            # Features that must be these set types
            if feature_name in self.get_categorical_features():
                continue

            if feature_name in self.get_bool_features():
                continue

            feature_values = set(pd.to_numeric(df[feature_name],
                                               errors="coerce").dropna())
            flag_dict = dict()

            # Bool check
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
            if flag_dict["Categorical"]:
                self.set_feature_to_categorical(feature_name)
                continue

            # -----
            if flag_dict["Numeric"]:
                if flag_dict["Float"]:
                    self.set_feature_to_float(feature_name)
                    continue

                elif flag_dict["Integer"]:
                    self.set_feature_to_integer(feature_name)
                    continue

        display(pd.DataFrame.from_dict(features_flag_types, orient='index'))


    def fix_string_features(self,
                            df,
                            notebook_mode=False):
        """
        Desc:
            Iterates through all string features and moves features to given types.
            May ask user question if it detects any conflicting string/numeric
            types.

        Args:
            df:
                Pandas dataframe object.

            notebook_mode:
                Will use the 'clear_output' notebook function if in notebook
                mode.
        """

        # Store types to convert
        type_conflict_dict = dict()

        # Currently this only performs
        for feature_name in self.get_string_features():

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
            for val, count in df[
                feature_name].dropna().value_counts().iteritems():

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

                    except ValueError:
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
                if user_input[0] == "s":
                    type_conflict_dict[feature_name] = "string"

                # -----
                elif user_input[0] == "n":
                    if float_flag:
                        type_conflict_dict[feature_name] = "float"
                    else:
                        numeric_values = pd.to_numeric(numeric_values,
                                                       errors="coerce")
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

            print(f"\nMoving feature '{feature_name}' to type {feature_type}")

            if feature_type == "string":
                self.set_feature_to_string(feature_name)

            elif feature_type == "datetime":
                self.set_feature_to_datetime(feature_name)

            elif feature_type == "integer":
                self.set_feature_to_integer(feature_name)

            elif feature_type == "category":
                self.set_feature_to_categorical(feature_name)

            elif feature_type == "bool":
                self.set_feature_to_bool(feature_name)

            elif feature_type == "float":
                self.set_feature_to_float(feature_name)

            else:
                raise TypeError("An unknown type was passed!")


    def fix_nan_features(self,
                         df):
        """
        Desc:
            Attempts to get the data type of the feature if there were no nans
            inside it.

        Args:
            df:
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
            directory_path:
                Absolute directory path.

            filename:
                File's given name
        """
        type_features = dict()
        for feature_name in self.get_bool_features():
            if "bool" in type_features.keys():
                type_features["bool"].append(feature_name)
            else:
                type_features["bool"] = [feature_name]

        for feature_name in self.get_string_features():
            if "string" in type_features.keys():
                type_features["string"].append(feature_name)
            else:
                type_features["string"] = [feature_name]

        for feature_name in self.get_integer_features():
            if "integer" in type_features.keys():
                type_features["integer"].append(feature_name)
            else:
                type_features["integer"] = [feature_name]

        for feature_name in self.get_float_features():
            if "float" in type_features.keys():
                type_features["float"].append(feature_name)
            else:
                type_features["float"] = [feature_name]

        for feature_name in self.get_categorical_features():
            if "categorical" in type_features.keys():
                type_features["categorical"].append(feature_name)
            else:
                type_features["categorical"] = [feature_name]

        for feature_name in self.get_datetime_features():
            if "datetime" in type_features.keys():
                type_features["datetime"].append(feature_name)
            else:
                type_features["datetime"] = [feature_name]

        type_features["target"] = self.__target_feature

        create_json_file_from_dict(type_features,
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
        self.__all_columns = set()

        self.__bool_features = set()
        self.__string_features = set()
        self.__categorical_features = set()
        self.__integer_features = set()
        self.__float_features = set()
        self.__datetime_features = set()

        self.__target_feature = None

        # Iterate through dict given types to feature lists
        for type, feature_names  in type_features.items():

            if type == "bool":
                self.__bool_features = set(feature_names)

            # -----
            elif type == "integer":
                self.__integer_features = set(feature_names)

            # -----
            elif type == "float":
                self.__float_features = set(feature_names)

            # -----
            elif type == "string":
                self.__string_features = set(feature_names)

            # -----
            elif type == "categorical":
                self.__categorical_features = set(feature_names)

            # -----
            elif type == "datetime":
                self.__datetime_features = set(feature_names)

            # Extract target
            elif type == "target":
                self.__target_feature = feature_names

            else:
                raise ValueError(f"Unknown type {type} was found!")



    def __bool_check(self,
                     feature_values):
        """
        Desc:
            Determines if the feature type of this column is a boolean based on
            the numeric values passed to it.

        Args:
            feature_values:
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
        # Categorical check
        if sum(feature_values) == sum(range(1, len(feature_values) + 1)):
            return True
        else:
            return False

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
        # -----
        float_check = False
        categorical_total_sum = 0
        categorical_check = True

        # Iterate through all values
        for val in feature_values:

            # Ignore None values
            if not val:
                continue
            try:
                float(val)
            except ValueError:
                return False, False, False, False

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
        if categorical_total_sum != sum(range(1, len(feature_values) + 1)):
            categorical_check = False

        return True, float_check, not float_check, categorical_check