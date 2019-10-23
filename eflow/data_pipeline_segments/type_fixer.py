from eflow.foundation import DataFrameTypes
from eflow._hidden.parent_objects import DataPipelineSegment
import numpy as np
import pandas as pd
from dateutil import parser
from IPython.display import display
from IPython.display import clear_output
import re

class TypeFixer(DataPipelineSegment):
    """
        Attempts to convert features to the correct types. Will update the
        dataframe and df_features.
    """
    def __init__(self,
                 segment_id=None):
        """
        Args:
            segment_id:
                Reference id to past segments of this object.

        Note/Caveats:
            When creating any public function that will be part of the pipeline's
            structure it is important to follow this given template. Also,
            try not to use _add_to_que. Can ruin the entire purpose of this
            project.
        """
        DataPipelineSegment.__init__(self,
                                     object_type=self.__class__.__name__,
                                     segment_id=segment_id)


    def numeric_feature_fix(self,
                            df,
                            df_features,
                            _add_to_que=True):
        """
        Desc:
            Attempts to convert numerical features to the correct types by
            following the given priority que:
                1. Bool
                2. Categorical
                3. Float
                4. Int
                5. Do nothing
            If the dataframe can't be updated to that given type do to nulls
            being present in the feature than the conversion will be ignored.
            However, df_features will be updated no matter what making it a
            great reference tool to see the actual types of the dataframe.

        Args:
            df:
                Pandas Dataframe object to update to correct types.

            df_features:
                DataFrameTypes object to update to correct types.

            _add_to_que:
                Pushes the function to pipeline segment parent if set to 'True'.
        """
        params_dict = locals()

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self", "df", "df_features", "_add_to_que",
                        "params_dict"]:
                del params_dict[arg]

        tmp_df_features = DataFrameTypes(df,
                                         ignore_nulls=True)

        features_flag_types = dict()
        for feature_name in df.columns:
            feature_values = set(df[feature_name].dropna())
            flag_dict = dict()

            # Ignore all string features
            if feature_name in tmp_df_features.get_string_features():
                continue

            # Features that must be these set types
            if feature_name in tmp_df_features.get_categorical_features():
                continue
            if feature_name in tmp_df_features.get_bool_features():
                continue

            # Bool check
            flag_dict["Bool"] = self.__bool_check(feature_values)
            flag_dict["Categorical"] = self.__categorical_check(feature_values)
            numeric_flag, float_flag, int_flag = self.__numeric_check(feature_values)
            flag_dict["Numeric"] = numeric_flag
            flag_dict["Float"] = float_flag
            flag_dict["Integer"] = int_flag

            # Pass the flag dictionary to later be processed by the priority que.
            features_flag_types[feature_name] = flag_dict

        # Iterate on feature and changes based on priority que
        for feature_name, flag_dict in features_flag_types.items():

            # -----
            if flag_dict["Bool"]:
                if not df[feature_name].isnull().values.any():
                    df[feature_name] = df[feature_name].astype('bool')

                df_features.set_feature_to_bool(feature_name)
                continue

            # -----
            if flag_dict["Categorical"]:
                if not df[feature_name].isnull().values.any():
                    df[feature_name] = df[feature_name].astype('category')
                df_features.set_feature_to_categorical(feature_name)
                continue

            # -----
            if flag_dict["Numeric"]:
                if flag_dict["Float"]:
                    df[feature_name] = df[feature_name].astype('float')
                    df_features.set_feature_to_float(feature_name)
                    continue

                elif flag_dict["Integer"]:
                    if not df[feature_name].isnull().values.any():
                        df[feature_name] = df[feature_name].astype('int')
                    df_features.set_feature_to_integer(feature_name)
                    continue

        display(pd.DataFrame.from_dict(features_flag_types, orient='index'
                                       ))
        if _add_to_que:
            self._DataPipelineSegment__add_function_to_que("numeric_feature_fix",
                                                           params_dict)


    def string_type_conflict_fix(self,
                                 df,
                                 df_features,
                                 type_conflict_dict,
                                 numeric_conflict_options,
                                 _add_to_que=True):
        params_dict = locals()

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self", "df", "df_features", "_add_to_que",
                        "params_dict"]:
                del params_dict[arg]

        for feature_name, feature_type in type_conflict_dict.items():

            converted_values = []
            for val in df[feature_name].values:
                if feature_type == "string" or feature_type == "datetime":
                    if isinstance(val, str):
                        converted_values.append(val)
                    else:
                        converted_values.append(np.nan)
                else:
                    if not isinstance(val, str):
                        converted_values.append(val)
                    else:
                        if feature_name in numeric_conflict_options:
                            if numeric_conflict_options[feature_name] == "convert to nans":
                                converted_values.append(np.nan)
                            elif numeric_conflict_options[feature_name] == "extract numeric values":
                                val = ''.join(char for char in str(val) if char.isdigit() or char == ".")

                                if len(val):
                                    if feature_type == "float":
                                        converted_values.append(float(val))

                                    elif feature_type == "integer":
                                        converted_values.append(int(val))

                                    else:
                                        raise ValueError(f"Unknown feature conversion to '{feature_type}' type.")

                                else:
                                    converted_values.append(np.nan)

                        else:
                            converted_values.append(np.nan)
            print(f"\nConverting feature '{feature_name}' to type {feature_type}")

            if feature_type == "string":
                df_features.set_feature_to_string(feature_name)

            elif feature_type == "datetime":
                df_features.set_feature_to_datetime(feature_name)

            elif feature_type == "integer":
                df_features.set_feature_to_integer(feature_name)

            elif feature_type == "float":
                df_features.set_feature_to_float(feature_name)
            else:
                raise TypeError("An unknown type was passed to this function")

            df[feature_name] = converted_values

        # Update df_features
        if _add_to_que:
            self._DataPipelineSegment__add_function_to_que("type_string_conflict_fix",
                                                           params_dict)


    def get_string_type_conflict_info(self,
                                     df,
                                     df_features,
                                     notebook_mode=False):
        type_conflict_dict = dict()
        numeric_conflict_options = dict()

        tmp_df_features = DataFrameTypes(df,
                                         ignore_nulls=False)

        # Currently this only performs
        for feature_name in tmp_df_features.get_string_features():

            float_flag = False

            numeric_count = 0
            numeric_values = []

            string_count = 0
            string_values = []

            datetime_count = 0
            datetime_values = []

            for val, count in df[feature_name].dropna().value_counts().iteritems():

                numeric_check = False

                try:
                    float(val)
                    numeric_check = True
                except ValueError:
                    pass

                if isinstance(val, float) or isinstance(val, int) or numeric_check == True:
                    numeric_values.append(val)
                    numeric_count += count

                    if isinstance(val, float):
                        float_flag = True

                    if numeric_check and isinstance(val,str):
                        if len(val.split(".")) == 2:
                            float_flag = True

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


            if numeric_count != 0 and string_count == 0 and datetime_count == 0:
                if float_flag:
                    type_conflict_dict[feature_name] = "float"
                else:
                    type_conflict_dict[feature_name] = "integer"

            elif numeric_count == 0 and string_count != 0 and datetime_count == 0:
                type_conflict_dict[feature_name] = "string"

            elif numeric_count == 0 and string_count == 0 and datetime_count != 0:
                type_conflict_dict[feature_name] = "datetime"

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
                    "\nConvert feature to numeric or string and replace any conflicts with nulls.\n* Numeric\n* String\n* Datetime\n* Ignore\nInput: ")
                user_input = user_input.lower()

                if user_input[0] == "s":
                    type_conflict_dict[feature_name] = "string"

                elif user_input[0] == "n":
                    if float_flag:
                        type_conflict_dict[feature_name] = "float"
                    else:
                        type_conflict_dict[feature_name] = "integer"

                    print("You can use the first character of the option for input.\n")
                    user_input = input(
                        "\nHow should unwanted values be handled?\n* Convert to Nans\n* Extract numeric values\nInput: ")

                    if user_input[0] == "e":
                        numeric_conflict_options[feature_name] = "extract numeric values"
                    else:
                        numeric_conflict_options[feature_name] = "convert to nans"

                    if notebook_mode:
                        clear_output()
                    else:
                        print()
                elif user_input[0] == "d":
                    type_conflict_dict[feature_name] = "datetime"

                else:
                    print(f"Ignoring feature '{feature_name}")

        return type_conflict_dict, numeric_conflict_options

    def __bool_check(self,
                     feature_values):

        if len(feature_values) == 2:
            return True
        else:
            return False

    def __categorical_check(self,
                            feature_values):
        # Categorical check
        last_val = None
        categorical_flag = True

        for val in feature_values:
            if not val:
                continue

            if not last_val:
                last_val = val
                continue
            try:
                if np.abs(val - last_val) != 1:
                    categorical_flag = False
                    break
            except:
                return False


            else:
                last_val = val
        return categorical_flag

    def __numeric_check(self,
                        feature_values):
        float_check = False
        for val in feature_values:

            if not val:
                continue

            try:
                float(val)
            except ValueError:
                return False, False, False

            val = str(val)
            tokens = val.split(".")
            if len(tokens) > 1 and int(tokens[1]) > 0:
                float_check = True

        return True, float_check, not float_check