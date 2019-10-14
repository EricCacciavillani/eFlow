from eflow._hidden.parent_objects import DataPipelineSegment
import copy
from collections import deque


class DataTransformer(DataPipelineSegment):
    def __init__(self,
                 segment_path_id=None):
        """
        project_sub_dir:
            Appends to the absolute directory of the output folder

        project_name:
            Creates a parent or "project" folder in which all sub-directories
            will be inner nested.

        overwrite_full_path:
            Overwrites the path to the parent folder.

        notebook_mode:
            If in a python notebook display visualizations in the notebook.
        """
        DataPipelineSegment.__init__(self,
                                     object_name=self.__class__.__name__)

    def remove_features(self,
                        df,
                        feature_names):

        if isinstance(feature_names, str):
            feature_names = [feature_names]

        for feature_n in feature_names:
            self.__check_if_feature_exists(df,
                                           feature_n)

        df.drop(columns=feature_names,
                inplace=True)
        self._DataPipelineSegment__add_function_to_que("remove_features",
                                                       feature_names)


    def __check_if_feature_exists(self,
                                  df,
                                  feature_name):
        if feature_name not in df.columns:
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the dataframe")

