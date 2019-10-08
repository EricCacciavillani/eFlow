# from eflow._hidden.custom_exceptions import UnsatisfiedRequirments

import copy
from collections import deque

class DataPipeline:
    def __init__(self,
                 pipeline_name):

        self.__pipeline_deque = deque()

    def add(self,
            pipeline_name,
            pipeline_obj):
        pipeline_name = copy.deepcopy(pipeline_name)
        pipeline_obj_name = pipeline_obj.__name__

        # if not isinstance(pipeline_obj,
        #                   DataPipelineSegment):
        #
        #     raise UnsatisfiedRequirments(f"Expected a 'DataPipelineSegment' object; received '{type(pipeline_obj)}'")

        self.__pipeline_deque.append()

        del pipeline_obj