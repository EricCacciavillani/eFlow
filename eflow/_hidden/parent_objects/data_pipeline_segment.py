from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.custom_exceptions import PipelineSegmentError
from eflow.utils.sys_utils import create_json_file_from_dict,get_all_files_from_path
from eflow.utils.string_utils import create_hex_decimal_string
from collections import deque
import copy
import os

class DataPipelineSegment(FileOutput):
    def __init__(self,
                 object_name):

        self.__object_name = object_name
        self.__function_pipe = deque()
        self.__pipeline_names = set()
        self.__json_file_name = None

    @property
    def file_path(self):
        if len(self.__function_pipe) == 0:
            raise PipelineSegmentError("The pipeline segment has not performed any actions yet."
                                       " Please perform some methods with this object.")
        else:
            return self.folder_path + copy.deepcopy(self.__json_file_name)

    @property
    def file_name(self):
        if len(self.__function_pipe) == 0:
            raise PipelineSegmentError("The pipeline segment has not performed any actions yet."
                                       " Please perform some methods with this object.")
        else:
            return copy.deepcopy(self.__json_file_name)


    def __add_function_to_que(self,
                              function_name,
                              param_vals):

        self.__function_pipe.append((function_name,
                                     param_vals))

        if len(self.__function_pipe) == 1:
            FileOutput.__init__(self,
                                f'_Extras/Data Pipeline Segments/{self.__object_name}')
            all_json_files = get_all_files_from_path(self.folder_path,
                                                     ".json")
            while True:
                random_file_name = create_hex_decimal_string().upper()
                if random_file_name not in all_json_files:
                    break
            self.__json_file_name = random_file_name

        self.__create_json_pipeline_segment_file()

    def __create_json_pipeline_segment_file(self,
                                            pipeline_name=None,
                                            pipeline_path=None):
        json_dict = dict()
        json_dict["Pipeline"] = dict()
        json_dict["Pipeline"]["Pipeline Name"] = pipeline_name
        json_dict["Pipeline"]["Pipeline Path"] = pipeline_path

        # Pipeline Segment
        json_dict["Pipeline Segment"] = dict()

        function_count = 1
        for function_name, params in self.__function_pipe:
            json_dict["Pipeline Segment"]["Object Type"] = self.__object_name
            json_dict["Pipeline Segment"]["Functions Performed Order"] = dict()
            json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function " \
                f"Order {function_count}"] = dict()
            json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function " \
                f"Order {function_count}"][function_name] = dict()
            json_dict["Pipeline Segment"]["Functions Performed Order"][
                f"Function Order {function_count}"][function_name][
                "Params"] = dict()
            for param_count,p in enumerate(params):
                json_dict["Pipeline Segment"]["Functions Performed Order"][
                    f"Function Order {function_count}"][function_name]["Params"][f"Param {param_count}"] = p

            function_count += 1


        create_json_file_from_dict(json_dict,
                                   self.folder_path,
                                   self.__json_file_name)

    def __check_create_pipeline_names(self,
                                      pipeline_name,
                                      pipeline_path):

        if pipeline_name in self.__pipeline_names:
            raise PipelineSegmentError("This object has already been pushed to"
                                       " a DataPipeline! Once pushed to a "
                                       "pipeline object this object can not"
                                       " be used again.")
        else:
            self.__pipeline_names.add(copy.deepcopy(pipeline_name))
            self.__create_json_pipeline_segment_file(pipeline_name,
                                                     pipeline_path)
