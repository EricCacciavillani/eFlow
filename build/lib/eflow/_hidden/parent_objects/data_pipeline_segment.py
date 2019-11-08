from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.custom_exceptions import PipelineSegmentError, UnsatisfiedRequirments
from eflow.utils.sys_utils import dict_to_json_file,json_file_to_dict,get_all_files_from_path, create_dir_structure
from eflow.utils.string_utils import create_hex_decimal_string
from collections import deque
import copy
import os
import inspect
import re

class DataPipelineSegment(FileOutput):
    def __init__(self,
                 object_type,
                 segment_id=None):

        self.__json_file_name = None
        self.__object_type = copy.deepcopy(object_type)
        if not isinstance(segment_id, str) and segment_id:
            raise UnsatisfiedRequirments(
                "Segment id must be a string or set to 'None'!")

        if isinstance(segment_id,str):
            segment_id = segment_id.split(".")[0]

        self.__segment_id = copy.deepcopy(segment_id)
        self.__function_pipe = deque()
        if self.__segment_id:
            self.__configure_pipeline_segment_with_existing_file()


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
                              params_dict):

        self.__function_pipe.append((function_name,
                                     params_dict))

        if len(self.__function_pipe) == 1 and not self.__json_file_name:
            FileOutput.__init__(self,
                                f'_Extras/Data Pipeline Segments/{self.__object_type}')
            all_json_files = get_all_files_from_path(self.folder_path,
                                                     ".json")
            while True:
                random_file_name = create_hex_decimal_string().upper()
                if random_file_name not in all_json_files:
                    break

            self.__segment_id = random_file_name
            self.__json_file_name = random_file_name + ".json"

        self.__create_json_pipeline_segment_file()

    def __create_json_pipeline_segment_file(self):
        json_dict = dict()
        # json_dict["Pipeline"] = dict()
        # json_dict["Pipeline"]["Pipeline Name"] = pipeline_name
        # json_dict["Pipeline"]["Pipeline Path"] = pipeline_name

        # Pipeline Segment
        json_dict["Pipeline Segment"] = dict()
        json_dict["Pipeline Segment"]["Object Type"] = self.__object_type
        json_dict["Pipeline Segment"]["Functions Performed Order"] = dict()
        function_order = 1
        for function_name, params_dict in self.__function_pipe:
            json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function " \
                f"Order {function_order}"] = dict()
            json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function " \
                f"Order {function_order}"][function_name] = dict()

            json_dict["Pipeline Segment"]["Functions Performed Order"][
                f"Function Order {function_order}"][function_name][
                "Params Dict"] = params_dict

            function_order += 1
        json_dict["Pipeline Segment"]["Function Count"] = function_order - 1

        dict_to_json_file(json_dict,
                                   self.folder_path,
                                   self.__json_file_name)

    def __configure_pipeline_segment_with_existing_file(self):

        FileOutput.__init__(self,
                            f'_Extras/Data Pipeline Segments/{self.__object_type}')

        self.__function_pipe = deque()
        self.__json_file_name = copy.deepcopy(self.__segment_id) + ".json"

        if not os.path.exists(self.folder_path):
            raise PipelineSegmentError(
                "Couldn't find the pipeline segment's folder when trying to configure this object with the provided json file.")
        if not os.path.exists(self.folder_path + copy.deepcopy(self.__json_file_name)):
            raise PipelineSegmentError(
                f"Couldn't find the pipeline segment's file named '{self.__json_file_name}' in the pipeline's directory when trying to configure this object with the provided json file.")

        json_dict = json_file_to_dict(
            self.folder_path + copy.deepcopy(self.__json_file_name))

        for function_order in range(1,2):
            function_name = list(json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function Order {function_order}"].keys())[0]
            params_dict = json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function Order {function_order}"][function_name]["Params Dict"]
            self.__function_pipe.append((function_name,
                                         params_dict))



    def perform_segment(self,
                        df):
        for function_name,params_dict in self.__function_pipe:
            method_to_call = getattr(self, function_name)

            params_dict["df"] = df
            params_dict["_add_to_que"] = False
            method_to_call(**params_dict)
            del params_dict["df"]
            del params_dict["_add_to_que"]

        for function_name, params_dict in self.__function_pipe:
            print(params_dict)


    def generate_code(self,
                      generate_file=True,
                      add_libs=True):

        if len(self.__function_pipe) == 0:
            raise PipelineSegmentError("Can't generate code when no methods of this segment have been used yet!")

        generated_code = []

        if add_libs:
            generated_code.append("from eflow.utils.math_utils import *")
            generated_code.append("from eflow.utils.image_processing_utils import *")
            generated_code.append("from eflow.utils.pandas_utils import *")
            generated_code.append("from eflow.utils.modeling_utils import *")
            generated_code.append("from eflow.utils.string_utils import *")
            generated_code.append("from eflow.utils.misc_utils import *")
            generated_code.append("from eflow.utils.sys_utils import *")
            generated_code.append("")

        for function_name, params_dict in self.__function_pipe:

            pre_made_code = inspect.getsource(getattr(self, function_name))

            first_lines_found = False
            def_start = False

            for parm,val in params_dict.items():
                generated_code.append(f"{parm} = {val}")

            for line in pre_made_code.split("\n"):

                if "def " in line:
                    def_start = True
                    continue

                if def_start:
                    if "):" in line:
                        def_start = False
                    continue

                if "params_dict" in line or "_add_to_que" in line:
                    continue

                if not re.search('[a-zA-Z]', line) and not first_lines_found:
                    continue
                else:
                    first_lines_found = True

                if "__add_function_to_que" in line:
                    continue

                generated_code.append(line.replace("        ", "", 1))

        generated_code.append("#" + "------"*5)

        if generate_file:
            create_dir_structure(self.folder_path,
                                       "Generated code")
            with open(self.folder_path + f'Generated code/{self.__segment_id}.py', 'r+') as filehandle:
                filehandle.truncate(0)
                for listitem in generated_code:
                    filehandle.write('%s\n' % listitem)
            print(f"Generated a python file named/at: {self.folder_path}Generated code/{self.__segment_id}.py")
        else:
            return generated_code