from eflow._hidden.parents_objects import DataPipelineSegment
from eflow._hidden.parents_objects import FileOutput
import copy
class FeatureTransformer(DataPipelineSegment):
    def __init__(self,
                 project_sub_dir="",
                 project_name="Feature Transformer",
                 overwrite_full_path=None,
                 notebook_mode=True):
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

        FileOutput.__init__(self,
                            f'{project_sub_dir}/{project_name}',
                            overwrite_full_path)
        self.__notebook_mode = copy.deepcopy(notebook_mode)
        pass


