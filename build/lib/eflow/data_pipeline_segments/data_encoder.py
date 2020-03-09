from eflow._hidden.parent_objects import DataPipelineSegment
from eflow._hidden.constants import BOOL_STRINGS
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow.utils.language_processing_utils import get_synonyms
from eflow.utils.misc_utils import get_parameters

import copy
import pandas as pd
import numpy as np

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class DataEncoder(DataPipelineSegment):
    """
        Attempts to convert features to the correct types. Will update the
        dataframe and df_features.
    """
    def __init__(self,
                 segment_id=None,
                 create_file=True):
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
                                     segment_id=segment_id,
                                     create_file=create_file)

    def encode_data(self,
                    df,
                    df_features,
                    apply_value_representation=True,
                    _add_to_que=True):
        """
        Desc:
            Encode the data into numerical values for machine learning processes.

        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            apply_value_representation: bool
                Translate features into most understandable/best representation/

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """

        # Apply value representation to feature values
        if apply_value_representation:
            feature_value_represention = df_features.get_feature_value_representation()

            # Inverse dict
            tmp_dict = copy.deepcopy(feature_value_represention)
            for feature_name in feature_value_represention.keys():
                tmp_dict[feature_name] = dict()
                for val, reprs in feature_value_represention[feature_name].items():
                    tmp_dict[feature_name][reprs] = val

            feature_value_represention = tmp_dict

            for feature_name in feature_value_represention.keys():
                if feature_name not in df.columns:
                    continue

                if df[feature_name].dtype == "O":
                    df[feature_name].replace(
                        feature_value_represention[feature_name],
                        inplace=True)

        # Decode data from categorical values to proper strings.
        encoder_dict = df_features.get_label_encoder()
        for feature_name in encoder_dict.keys():

            if feature_name not in df.columns:
                continue

            if df[feature_name].dtype == "O":
                df[feature_name].replace(encoder_dict[feature_name],
                                         inplace=True)

                df_features.set_feature_to_categorical(feature_name)

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.encode_data)
            self._DataPipelineSegment__add_function_to_que("encode_data",
                                                           parameters,
                                                           params_dict)

    def decode_data(self,
                    df,
                    df_features,
                    apply_value_representation=True,
                    _add_to_que=True):
        """
        Desc:
            Decode the data into non-numerical values for more descriptive analysis.

        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            apply_value_representation: bool
                Translate features into most understandable/best representation/

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """
        # Decode data from categorical values to proper strings.
        decoder_dict = df_features.get_label_decoder()
        for feature_name in decoder_dict.keys():

            if feature_name not in df.columns:
                continue

            if df[feature_name].dtype != "O":
                df[feature_name].replace(decoder_dict[feature_name],
                                         inplace=True)

        # Apply value representation to feature values
        if apply_value_representation:
            feature_value_represention = df_features.get_feature_value_representation()
            # Replace values by each corresponding feature value related dict
            for feature_name in feature_value_represention.keys():
                if feature_name not in df.columns:
                    continue

                if df[feature_name].dtype == "O":
                    df[feature_name].replace(feature_value_represention[feature_name],
                                             inplace=True)

                df_features.set_feature_to_string(feature_name)

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.decode_data)
            self._DataPipelineSegment__add_function_to_que("decode_data",
                                                           parameters,
                                                           params_dict)

    def apply_binning(self,
                      df,
                      df_features,
                      binable_features=[],
                      _add_to_que=True):

        # Remove any unwanted arguments in params_dict
        params_dict = locals()
        for arg in ["self", "df", "df_features", "_add_to_que",
                    "params_dict"]:
            try:
                del params_dict[arg]
            except KeyError:
                pass

        if not binable_features:
            binable_features = df.columns

        # Apply binning
        for feature_name in binable_features:
            bin_labels_dict = df_features.get_feature_binning(feature_name)
            if bin_labels_dict:
                # Convert to category data
                df[feature_name] = pd.to_numeric(df[feature_name].dropna(),
                                                 errors='coerce')
                df[feature_name] = pd.cut(df[feature_name],
                                          bins=bin_labels_dict["bins"],
                                          labels=bin_labels_dict["labels"])

                # Feature set to categorical
                df_features.set_feature_to_categorical(feature_name)

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.decode_data)
            self._DataPipelineSegment__add_function_to_que("apply_binning",
                                                           parameters,
                                                           params_dict)

    def apply_value_representation(self,
                                   df,
                                   df_features,
                                   _add_to_que=True):
        """
        Desc:
            Translate features into most understandable/best representation

        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """
        feature_value_represention = df_features.get_feature_value_representation()

        # Replace values by each corresponding feature value related dict
        for feature_name in feature_value_represention:

            if feature_name not in df.columns:
                raise KeyError(
                    f"Dataframe doesn't have feature name '{feature_name}'.")

            df[feature_name].replace(feature_value_represention[feature_name],
                                     inplace=True)

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.decode_data)
            self._DataPipelineSegment__add_function_to_que("apply_value_representation",
                                                           parameters,
                                                           params_dict)

    def revert_value_representation(self,
                                    df,
                                    df_features,
                                    _add_to_que=True):
        """
        Desc:
            Translate features back into worst representation

        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """
        feature_value_represention = df_features.get_feature_value_representation()

        # Replace values by each corresponding feature value related dict
        for feature_name in feature_value_represention:

            if feature_name not in df.columns:
                raise KeyError(
                    f"Dataframe doesn't have feature name '{feature_name}'.")


            df[feature_name].replace({v: k
                                      for k, v in feature_value_represention[
                                          feature_name].items()},
                                     inplace=True)

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.revert_value_representation)
            self._DataPipelineSegment__add_function_to_que("revert_value_representation",
                                                           parameters,
                                                           params_dict)

    def make_values_bool(self,
                         df,
                         df_features,
                         _add_to_que=True):
        """
        Desc:
            Convert all string bools to numeric bool value
        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """
        for bool_feature in df_features.bool_features():
            if df[bool_feature].dtype == "O":
                bool_check,true_val,false_val = self.__bool_string_values_check(
                    df[bool_feature].dropna().unique())

                # Replace bool string values with bools
                if bool_check:
                    df[bool_feature].replace({true_val:1,
                                              false_val:0},
                                             inplace=True)
        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.make_values_bool)
            self._DataPipelineSegment__add_function_to_que("make_values_bool",
                                                           parameters,
                                                           params_dict)

    def make_dummies(self,
                     df,
                     df_features,
                     qualitative_features=[],
                     _feature_values_dict=None,
                     _add_to_que=True):
        """
        Desc:
            Create dummies features of based on qualtative feature data and removes
            the original feature.

            Note
                _feature_values_dict does not need to be init. Used for backend
                resource.

        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            qualtative_features: collection of strings
                Feature names to convert the feature data into dummy features.

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """
        # Convert to the correct types
        if isinstance(qualitative_features,str):
            qualtative_features = [qualitative_features]

        if not _feature_values_dict:
            _feature_values_dict = dict()

        pd.set_option('mode.chained_assignment', None)

        for cat_feature in qualitative_features:

            if cat_feature not in df_features.string_features() | df_features.categorical_features():
                raise UnsatisfiedRequirments(f"No feature named '{cat_feature}' in categorical or string features.")

            if cat_feature not in _feature_values_dict:
                _feature_values_dict[cat_feature] = df[cat_feature].dropna().unique()
                _feature_values_dict[cat_feature].sort()
                _feature_values_dict[cat_feature] = _feature_values_dict[cat_feature].tolist()

            dummy_features = []
            for feature_value in  _feature_values_dict[cat_feature]:
                new_feature = cat_feature + f"_{feature_value}"
                bool_array = df[cat_feature] == feature_value
                df[new_feature] = copy.deepcopy(bool_array)
                dummy_features.append(new_feature)

            # # Make dummies and remove original feature
            # dummies_df = pd.get_dummies(_feature_values_dict[cat_feature],
            #                             prefix=cat_feature)

            df.drop(columns=[cat_feature],
                    inplace=True)
            df_features.remove_feature(cat_feature)
            df_features.set_feature_to_dummy_encoded(cat_feature,
                                                     dummy_features)

            # # Apply to dataframe
            # for feature_name in dummies_df.columns:
            #     df[feature_name] = dummies_df[feature_name]

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.make_dummies)
            self._DataPipelineSegment__add_function_to_que("make_dummies",
                                                           parameters,
                                                           params_dict)

    def revert_dummies(self,
                       df,
                       df_features,
                       qualitative_features=[],
                       _add_to_que=True):
        """
        Desc:
            Convert dummies features back to the original feature.

        Args:
            df: pd.Dataframe
                Pandas dataframe.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            qualitative_features: collection of strings
                Feature names to convert the dummy features into original feature
                data.

            _add_to_que: bool
                Hidden variable to determine if the function should be pushed
                to the pipeline segment.
        """

        df.reset_index(inplace=True,
                       drop=True)

        if isinstance(qualitative_features, str):
            feature_name = [qualitative_features]

        for feature_name in qualitative_features:
            dummies_df = df[
                df_features.get_dummy_encoded_features()[feature_name]]
            dummies_columns = dummies_df.columns.to_list()

            tmp_df = dummies_df[dummies_df == 1].stack().reset_index()
            del dummies_df

            df[feature_name] = np.full([len(df)], np.nan)
            df[feature_name].iloc[tmp_df["level_0"]] = tmp_df[
                "level_1"].values.tolist()

            # Remove dummy features
            df.drop(columns=dummies_columns,
                    inplace=True)

            df[feature_name] = df[feature_name].str[len(feature_name) + 1:]

            # Remove dummy encoded relationship
            df_features.remove_feature_from_dummy_encoded(feature_name)

            # Add feature back to original set in df_features
            try:
                pd.to_numeric(df[feature_name].dropna())
                df_features.add_new_categorical_feature(feature_name)

            except ValueError:
                df_features.add_new_string_feature(feature_name)

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.revert_dummies)
            self._DataPipelineSegment__add_function_to_que("revert_dummies",
                                                           parameters,
                                                           params_dict)


    def __bool_string_values_check(self,
                                   feature_values):
        """
        Desc:
            Checks if a collection of strings can be considered a bool feature
            based on the amount of strings and the values of those strings.

            Note -
                Modified from data frame types
        Args:
            feature_values: collection
                Collection of strings to apply natural language process to
                determine if the series data is boolean or not.

        Returns:
            Returns true or false if the values can be considered a bool and
            the true and false values found.
        """

        if len(feature_values) > 2:
            return False, None, None

        found_true_value = None
        found_false_value = None

        for val in feature_values:

            if not isinstance(val,str):
                continue

            org_val = copy.deepcopy(val)
            val = val.lower()

            # Determine if val is true
            if not found_true_value:

                # Check if the string already exist in the defined set
                if val in BOOL_STRINGS.TRUE_STRINGS:
                    found_true_value = org_val
                    continue
                else:
                    # Attempt to find synonyms of the defined words to compare to
                    # the iterable string
                    for true_string in BOOL_STRINGS.TRUE_STRINGS:

                        if len(true_string) < 2:
                            continue

                        for syn in get_synonyms(true_string):
                            if syn == val:
                                found_true_value = org_val
                                continue

            # -----
            if not found_false_value:

                # -----
                if val in BOOL_STRINGS.FALSE_STRINGS:
                    found_false_value = org_val
                    continue
                else:
                    # -----
                    for false_string in BOOL_STRINGS.FALSE_STRINGS:

                        if len(false_string) < 2:
                            continue

                        for syn in get_synonyms(false_string):
                            if syn == val:
                                found_false_value = org_val
                                continue

        if len(feature_values) == 2:
            return isinstance(found_true_value,str) and isinstance(found_false_value,str), found_true_value, found_false_value
        else:
            return isinstance(found_true_value,str) or isinstance(found_false_value,str), found_true_value, found_false_value
