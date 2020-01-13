from eflow._hidden.parent_objects import DataPipelineSegment
from eflow.utils.pandas_utils import check_if_feature_exists
from eflow.utils.misc_utils import get_parameters

import copy

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

# Template
# def METHOD_NAME(self,
#                 df,
#                 df_features,
#                 '''ALL YOUR OTHER ARGS'''
#                 _add_to_que=True):

#
#     '''
#     YOUR CUSTOM CODE HERE
#     '''
#
#
#     # Add to the given pipeline segment
#     if _add_to_que:
#         params_dict = locals()
#          parameters = get_parameters(self.METHOD_NAME)
#          self._DataPipelineSegment__add_function_to_que("METHOD_NAME",
#                                                         parameters,
#                                                         params_dict)


class FeatureTransformer(DataPipelineSegment):
    """
        Combines, removes, scales, etc features of a pandas dataframe.
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

    def remove_features(self,
                        df,
                        df_features,
                        feature_names,
                        _add_to_que=True):
        """
        Desc:
            Removes unwanted features from the dataframe and saves them to the
            pipeline segment structure if _add_to_que is set to True.

        Args:
            df:
                Pandas Dataframe to update.

            df_features:
                DataFrameTypes object to update.

            feature_names:
                Features to remove

            _add_to_que:
                Pushes the function to pipeline segment parent if set to 'True'.
        """

        if isinstance(feature_names, str):
            feature_names = [feature_names]

        for feature_n in feature_names:

            try:

                if feature_n in df_features.all_features():
                    df_features.remove_feature(feature_n)

                check_if_feature_exists(df,
                                        feature_n)
                df.drop(columns=[feature_n],
                        inplace=True)

            except KeyError:
                pass

        if _add_to_que:
            params_dict = locals()
            parameters = get_parameters(self.remove_features)
            self._DataPipelineSegment__add_function_to_que("remove_features",
                                                           parameters,
                                                           params_dict)