from eflow._hidden.parent_objects import DataPipelineSegment

import copy

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class TypeFixer(DataPipelineSegment):
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

    def convert_types(self,
                      df,
                      df_features,
                      user_logic=True):
        pass