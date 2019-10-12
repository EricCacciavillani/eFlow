from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.parent_objects import DataPipelineSegment
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, PipelineError
from eflow.utils.string_utils import create_hex_decimal_string
import copy
from collections import deque

class DataPipeline(FileOutput):
    def __init__(self,
                 pipeline_name):
        self.__pipeline_name = copy.deepcopy(pipeline_name)
        self.__pipeline_deque = deque()
        self.__pipeline_segment_labels = set()
        FileOutput.__init__(self,
                            project_name='_Extras/Data Pipelines')


    def add(self,
            label,
            pipeline_segment_obj):
        pipeline_obj_name = pipeline_segment_obj.__class__.__name__

        if not isinstance(pipeline_segment_obj,
                          DataPipelineSegment):

            raise UnsatisfiedRequirments(f"Expected a 'DataPipelineSegment' object; received '{type(pipeline_segment_obj)}'")

        pipeline_segment_obj._DataPipelineSegment__check_create_pipeline_names(
            self.__pipeline_name,
            self.relative_folder_path)

        if label in self.__pipeline_segment_labels:
            raise PipelineError(f"The '{label}' pipeline segment is already in this pipeline")

        self.__pipeline_segment_labels.add(label)

        # Create JSON file


        # ----------------
        self.__pipeline_deque.append((pipeline_obj_name,
                                      pipeline_segment_obj))
        self.__create_json_pipeline_file(pipeline_segment_obj)


    def __create_json_pipeline_file(self,
                                    pipeline_segment_obj):
        json_dict = dict()
        segment_count = 1

        for pipeline_obj_name, pipeline_obj in self.__pipeline_deque:
            json_dict[f"Pipeline Segment Order {segment_count}"] = dict()
            json_dict[f"Pipeline Segment Order {segment_count}"]["Order"] = segment_count
            json_dict[f"Pipeline Segment Order {segment_count}"]["Pipeline Segment Path"] = pipeline_segment_obj.relative_folder_path

            segment_count += 1
