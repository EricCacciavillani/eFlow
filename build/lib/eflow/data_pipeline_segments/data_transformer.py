from eflow._hidden.parent_objects import DataPipelineSegment
from eflow.utils.pandas_utils import check_if_feature_exists

import copy
from collections import deque


class DataTransformer(DataPipelineSegment):
    def __init__(self,
                 segment_id=None):
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

        Note/Caveats:
            When creating any public function that will be part of the pipeline's
            structure it is important to follow this given template. Also please never
            touch _add_to_que...ruins the entire purpose of this project.
            I needed to put it there to make the project come together.

            def your_code(self,
                          df,
                          '''ALL YOUR OTHER ARGS'''
                          _add_to_que=True):

                params_dict = locals()

                # Remove any unwanted arguments in params_dict
                if _add_to_que:
                    params_dict = locals()
                    for arg in ["self","df","_add_to_que", "params_dict"]:
                        del params_dict[arg]

                '''
                YOUR CUSTOM CODE HERE
                '''

                # Add to the given pipeline segment
                 if _add_to_que:
                    self._DataPipelineSegment__add_function_to_que("remove_features",
                                                                   params_dict)
        """
        DataPipelineSegment.__init__(self,
                                     object_type=self.__class__.__name__,
                                     segment_id=segment_id)

    def remove_features(self,
                        df,
                        feature_names,
                        _add_to_que=True):
        params_dict = locals()

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self","df","_add_to_que", "params_dict"]:
                del params_dict[arg]


        if isinstance(feature_names, str):
            feature_names = [feature_names]

        for feature_n in feature_names:
            check_if_feature_exists(df,
                                    feature_n)
            df.drop(columns=[feature_n],
                    inplace=True)

        if _add_to_que:
            self._DataPipelineSegment__add_function_to_que("remove_features",
                                                           params_dict)