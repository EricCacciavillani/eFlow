from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.parent_objects import DataPipelineSegment
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, PipelineError
from eflow.utils.string_utils import create_hex_decimal_string,correct_directory_path
from eflow.utils.sys_utils import create_json_file_from_dict, get_all_files_from_path, check_create_dir_structure, json_file_to_dict
from eflow._hidden.constants import SYS_CONSTANTS
import copy
import os
from collections import deque

class DataPipeline(FileOutput):
    def __init__(self,
                 pipeline_name,
                 pipeline_modify_id=None):
        dir_path_to_pipeline = correct_directory_path(f"{os.getcwd()}/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/_Extras/JSON Files/Data Pipeline/{pipeline_name}")
        configure_existing_file = False

        # Get json proper file name
        if pipeline_modify_id:
            json_file = f"{pipeline_modify_id.split('.')[0]}.json"
        else:
            json_file = "root_pipeline.json"

        # Check if folder/file exist for the pipeline
        if os.path.exists(dir_path_to_pipeline):
            if os.path.exists(dir_path_to_pipeline + json_file):
                print(f"The file '{json_file}' exist!")
                FileOutput.__init__(self,
                                    f'_Extras/JSON Files/Data Pipeline/{pipeline_name}')
                configure_existing_file = True
            else:
                raise PipelineError(f"The file '{json_file}' does not exist!")


        # Check if root file exist or if pipeline modify id
        self.__pipeline_name = copy.deepcopy(pipeline_name)
        self.__pipeline_segment_deque = deque()
        self.__pipeline_segment_names = set()
        self.__pipeline_segment_path_id = set()
        self.__pipeline_modify_id = copy.deepcopy(pipeline_modify_id)

        self.__json_file_name = json_file

        if configure_existing_file:
            print("Now configuring object with proper pipeline segments...")
            self.__configure_pipeline_with_existing_file()

    @property
    def file_path(self):
        if len(self.__pipeline_segment_deque) == 0:
            raise PipelineError(
                "The pipeline has not added any segments yet."
                " Please add segments to this object.")
        else:
            return self.folder_path + copy.deepcopy(self.__json_file_name)

    @property
    def file_name(self):
        if len(self.__pipeline_segment_deque) == 0:
            raise PipelineError(
                "The pipeline has not added any segments yet."
                " Please add segments to this object.")
        else:
            return copy.deepcopy(self.__json_file_name)


    def add(self,
            segment_name,
            pipeline_segment_obj):

        if not isinstance(pipeline_segment_obj,
                          DataPipelineSegment):

            raise UnsatisfiedRequirments(f"Expected a 'DataPipelineSegment' object; received '{type(pipeline_segment_obj)}'")

        if segment_name in self.__pipeline_segment_names:
            raise PipelineError(f"The '{segment_name}' pipeline segment is already in this pipeline. Please choose a different segment name.")

        segment_path_id = (pipeline_segment_obj.relative_folder_path + pipeline_segment_obj.file_name)
        if segment_path_id in self.__pipeline_segment_path_id:
            raise PipelineError("The segment has been already found "
                                "in this pipeline Segment path id: " +
                                f"'{segment_path_id}.'\n" +
                                "This can be done by:"
                                "\n\t*Creating a completely new segment object "
                                "and adding it to the pipeline with the 'add'"
                                " method."
                                "\n\t*Refer to a different segment path id")
        else:
            if len(self.__pipeline_segment_deque) == 0:
                FileOutput.__init__(self,
                                    f'_Extras/JSON Files/Data Pipeline/{self.__pipeline_name}')



        self.__pipeline_segment_names.add(segment_name)
        self.__pipeline_segment_path_id.add(segment_path_id)
        self.__pipeline_segment_deque.append((segment_name,
                                              pipeline_segment_obj))

        # pipeline_segment_obj._DataPipelineSegment__create_json_pipeline_segment_file()

        self.__create_json_pipeline_file()


    def __create_json_pipeline_file(self):

        json_dict = dict()
        segment_order = 1

        for segment_name, pipeline_segment_obj in self.__pipeline_segment_deque:
            json_dict["Pipeline Name"] = self.__pipeline_name
            json_dict["Pipeline Segment Order"] = dict()
            json_dict["Pipeline Segment Order"][segment_order] = dict()
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Path"] = pipeline_segment_obj.relative_folder_path + pipeline_segment_obj.file_name
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Type"] = pipeline_segment_obj.__class__.__name__
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Name"] = segment_name

            segment_order += 1

        json_dict["Pipeline Segment Count"] = segment_order
        if self.__pipeline_modify_id:
            check_create_dir_structure(self.folder_path,
                                       "/Modified Pipelines")
            create_json_file_from_dict(json_dict,
                                       self.folder_path + "/Modified Pipelines",
                                       self.__json_file_name)
        else:
            create_json_file_from_dict(json_dict,
                                       self.folder_path,
                                       self.__json_file_name)

    def __configure_pipeline_with_existing_file(self):
        self.__pipeline_segment_deque = deque()
        self.__pipeline_segment_names = set()
        self.__pipeline_segment_path_id = set()
        json_dict = json_file_to_dict(self.folder_path + copy.deepcopy(self.__json_file_name))

        for segment_order in range(1,json_dict["Pipeline Segment Count"]):
            # self.__pipeline_segment_names.add(segment_name)
            # self.__pipeline_segment_path_id.add(segment_path_id)
            # self.__pipeline_segment_deque.append((segment_name,
            #                                       pipeline_segment_obj))
            pass


    def perform_pipeline(self):
        json_dict = json_file_to_dict(self.file_path)

        for segment_order in range(1, json_dict["Pipeline Segment Count"]):
            pass

            # execute_str = f'segment_obj = {json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Type"]}"()''
            # execute_str += "segment_obj."

            # exec ("a=4\na+=1\nprint(a)")


