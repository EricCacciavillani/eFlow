from eflow._hidden.parents_objects import DataPipelineSegment
import copy
from collections import deque


class DataTransformer(DataPipelineSegment):
    def __init__(self):
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
                                     project_name=f'_Extras/Data Pipelines/FeatureTransformer')

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
        try:
            if feature_name:
                df[feature_name]
        except KeyError:
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the dataframe")
