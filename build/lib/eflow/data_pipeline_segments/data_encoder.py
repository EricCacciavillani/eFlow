from eflow._hidden.parent_objects import DataPipelineSegment

import copy

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

    def encode_data(self,
                    df,
                    df_features,
                    apply_value_representation=True,
                    _add_to_que=True):
        params_dict = locals()

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self", "df", "df_features", "_add_to_que",
                        "params_dict"]:
                del params_dict[arg]

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
                    raise KeyError(
                        f"Dataframe doesn't have feature name '{feature_name}'.")

                if df[feature_name].dtype == "O":
                    df[feature_name].replace(
                        feature_value_represention[feature_name],
                        inplace=True)

        # Decode data from categorical values to proper strings.
        encoder_dict = df_features.get_label_encoder()
        for feature_name in encoder_dict.keys():

            if feature_name not in df.columns:
                raise KeyError(
                    f"Dataframe doesn't have feature name '{feature_name}'.")

            if df[feature_name].dtype == "O":
                df[feature_name].replace(encoder_dict[feature_name],
                                         inplace=True)

        if _add_to_que:
            self._DataPipelineSegment__add_function_to_que("encode_data",
                                                           params_dict)

    def decode_data(self,
                    df,
                    df_features,
                    apply_value_representation=True,
                    _add_to_que=True):
        params_dict = locals()

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self", "df", "df_features", "_add_to_que",
                        "params_dict"]:
                del params_dict[arg]

        # Decode data from categorical values to proper strings.
        decoder_dict = df_features.get_label_decoder()
        for feature_name in decoder_dict.keys():

            if feature_name not in df.columns:
                raise KeyError(
                    f"Dataframe doesn't have feature name '{feature_name}'.")

            if df[feature_name].dtype != "O":
                df[feature_name].replace(decoder_dict[feature_name],
                                         inplace=True)

        # Apply value representation to feature values
        if apply_value_representation:
            feature_value_represention = df_features.get_feature_value_representation()
            for feature_name in feature_value_represention.keys():
                if feature_name not in df.columns:
                    raise KeyError(
                        f"Dataframe doesn't have feature name '{feature_name}'.")

                if df[feature_name].dtype == "O":
                    df[feature_name].replace(feature_value_represention[feature_name],
                                             inplace=True)

        if _add_to_que:
            self._DataPipelineSegment__add_function_to_que("decode_data",
                                                           params_dict)


