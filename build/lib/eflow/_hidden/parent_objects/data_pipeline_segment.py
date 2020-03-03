from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.custom_exceptions import PipelineSegmentError, UnsatisfiedRequirments
from eflow.utils.sys_utils import dict_to_json_file,json_file_to_dict,get_all_files_from_path, create_dir_structure
from eflow.utils.string_utils import create_hex_decimal_string

from collections import deque
import copy
import os
import inspect
import re

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class DataPipelineSegment(FileOutput):
    """
        Holds the function name's and arguments to be pushed to a json file.
        This generated json file later allows a dataframe to be passed to it
        and applies the functions and arguments in the sequence they were run.
    """
    def __init__(self,
                 object_type,
                 segment_id=None,
                 create_file=True):
        """
        Args:
            object_type: string
                The child type of all object's that inherited DataPipelineSegment

            segment_id: string
                 If init as a string instead of None; the object will attempt
                 to find the json file in the provided directory.
        Note:
            Essentially we are serializing the object with json files.
        """

        self.__json_file_name = None
        self.__object_type = copy.deepcopy(object_type)

        if not isinstance(segment_id, str) and segment_id:
            raise UnsatisfiedRequirments(
                "Segment id must be a string or set to 'None'!")

        if segment_id and not create_file:
            raise PipelineSegmentError("Parameter conflict: segment_id is referring "
                                       "to a saved file but create_file is set to False.")

        # File extension removal
        if isinstance(segment_id,str):
            segment_id = segment_id.split(".")[0]
        self.__segment_id = copy.deepcopy(segment_id)

        # Pushes the functions info based on order they are called
        self.__function_pipe = deque()

        self.__create_file = create_file
        self.__lock_interaction = False

        # Attempt to get json file into object's attributes.
        if self.__segment_id:
            self.__configure_pipeline_segment_with_existing_file()


    def perform_segment(self,
                        df,
                        df_features):
        """
        Desc:
            Performs all functions that the child of the pipeline segment has
            performed on a given pandas dataframe.

        Args:
            df: pd.Dataframe
                Pandas Dataframe

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.
        """
        for function_name, params_dict in self.__function_pipe:

            # Get function
            method_to_call = getattr(self, function_name)

            # -----
            params_dict["df"] = df
            params_dict["df_features"] = df_features
            params_dict["_add_to_que"] = False

            # Function call with arguments
            method_to_call(**params_dict)

            # -----
            del params_dict["df"]
            del params_dict["df_features"]
            del params_dict["_add_to_que"]

    # def generate_code(self,
    #                   generate_file=True,
    #                   add_libs=True):
    #     """
    #     Desc:
    #         Attempts to parse the file of the child object of the pipline
    #         segment.
    #
    #     Args:
    #         generate_file:
    #             Depending on the boolean value of generate_file; True will
    #             generate a python file and False will just return a list of
    #             strings.
    #
    #         add_libs:
    #             Adds utils libs to the top of list.
    #
    #     Returns:
    #         If the arg 'generate_file' is set to False then it will just return
    #         a list of strings.
    #     """
    #
    #     # Raise error if no methods of the child have been called yet
    #     if len(self.__function_pipe) == 0:
    #         raise PipelineSegmentError(
    #             "Can't generate code when no methods of this segment have been used yet!")
    #
    #     generated_code = []
    #
    #     # Add libs to the top of the file
    #     if add_libs:
    #         generated_code.append("from eflow.utils.math_utils import *")
    #         generated_code.append("from eflow.utils.image_processing_utils import *")
    #         generated_code.append("from eflow.utils.pandas_utils import *")
    #         generated_code.append("from eflow.utils.modeling_utils import *")
    #         generated_code.append("from eflow.utils.string_utils import *")
    #         generated_code.append("from eflow.utils.misc_utils import *")
    #         generated_code.append("from eflow.utils.sys_utils import *")
    #         generated_code.append("")
    #
    #     # -----
    #     for function_name, params_dict in self.__function_pipe:
    #
    #         # Get function's code
    #         pre_made_code = inspect.getsource(getattr(self, function_name))
    #
    #         first_lines_found = False
    #         def_start = False
    #
    #         # Init variables
    #         for parm, val in params_dict.items():
    #             generated_code.append(f"{parm} = {val}")
    #
    #         # Iterate line by line of function's code
    #         for line in pre_made_code.split("\n"):
    #
    #             if "def " in line:
    #                 def_start = True
    #                 continue
    #
    #             if def_start:
    #                 if "):" in line:
    #                     def_start = False
    #                 continue
    #
    #             if "params_dict" in line or "_add_to_que" in line:
    #                 continue
    #
    #             if not re.search('[a-zA-Z]', line) and not first_lines_found:
    #                 continue
    #             else:
    #                 first_lines_found = True
    #
    #             if "__add_function_to_que" in line:
    #                 continue
    #
    #             # -----
    #             generated_code.append(line.replace("        ", "", 1))
    #
    #         # Formatting
    #         generated_code.append("# " + "------" * 5)
    #
    #     # Generate file or pass back list
    #     if generate_file:
    #         create_dir_structure(self.folder_path,
    #                                    "Generated code")
    #         with open(
    #                 self.folder_path + f'Generated code/{self.__segment_id}.py',
    #                 'r+') as filehandle:
    #             filehandle.truncate(0)
    #             for listitem in generated_code:
    #                 filehandle.write('%s\n' % listitem)
    #         print(
    #             f"Generated a python file named/at: {self.folder_path}Generated code/{self.__segment_id}.py")
    #     else:
    #         return generated_code

    def reset_segment_file(self):
        # File/Folder error checks
        if not os.path.exists(self.folder_path):
            raise PipelineSegmentError(
                "Couldn't find the pipeline segment's folder when trying to configure this object with the provided json file.")
        if not os.path.exists(
                self.folder_path + copy.deepcopy(self.__json_file_name)):
            raise PipelineSegmentError(
                f"Couldn't find the pipeline segment's file named '{self.__json_file_name}' in the pipeline's directory when trying to configure this object with the provided json file.")

        dict_to_json_file({},
                          self.folder_path,
                          self.file_name)

    @property
    def file_path(self):
        """
        Desc:
            Gets the absolute path to the json file.
        """
        # -----
        if len(self.__function_pipe) == 0:
            raise PipelineSegmentError("The pipeline segment has not performed any actions yet."
                                       " Please perform some methods with this object.")
        elif not self.__create_file:
            raise PipelineSegmentError("This pipeline segment does not have saved "
                                       "file and thus can not have a file path.")
        else:
            return self.folder_path + copy.deepcopy(self.__json_file_name)

    @property
    def file_name(self):
        """
        Desc:
            Gets the file name of the json file.
        """
        # -----
        if len(self.__function_pipe) == 0:
            raise PipelineSegmentError("The pipeline segment has not performed any actions yet."
                                       " Please perform some methods with this object.")
        elif not self.__create_file:
            raise PipelineSegmentError("This pipeline segment does not have saved "
                                       "file and thus can not have a file path.")
        else:
            return copy.deepcopy(self.__json_file_name)


    def __replace_function_in_que(self,
                                  function_name,
                                  params_dict,
                                  param,
                                  param_val):

        raise ValueError("This function hasn't been completed yet!")

        if self.__lock_interaction:
            raise PipelineSegmentError("This pipeline has be locked down and "
                                       "will prevent futher changes to the generated flat file.")

        for delete_key in {"self", "df", "df_features", "_add_to_que",
                           "params_dict"}:
            if delete_key in params_dict.keys():
                del params_dict[delete_key]

        for k, v in {k: v for k, v in params_dict.items()}.items():
            if k not in parameters:
                del params_dict[k]
            elif isinstance(v, set):
                params_dict[k] = list(v)

        self.__function_pipe.append((function_name,
                                     params_dict))

        # Generate new json file name with proper file/folder output attributes
        if len(self.__function_pipe) == 1 and not self.__json_file_name:
            FileOutput.__init__(self,
                                f'_Extras/Pipeline Structure/Data Pipeline Segments/{self.__object_type}')
            all_json_files = get_all_files_from_path(self.folder_path,
                                                     ".json")
            while True:
                random_file_name = create_hex_decimal_string().upper()
                if random_file_name not in all_json_files:
                    break

            self.__segment_id = random_file_name
            self.__json_file_name = random_file_name + ".json"

        # Update json file
        if self.__create_file:
            self.__create_json_pipeline_segment_file()


    def __add_function_to_que(self,
                              function_name,
                              parameters,
                              params_dict):
        """
        Desc:
            Adds the function info the function que. If the segment has no
            json file name then generate one for it the given directory.

        Args:
            function_name: string
                Functions name

            params_dict: dict
                Parameter's name to their associated values.

        Note:
            This function should only ever be called by children of
            this object.
        """
        if self.__lock_interaction:
            raise PipelineSegmentError("This pipeline has be locked down and "
                                       "will prevent futher changes to the generated flat file.")


        for delete_key in {"self", "df", "df_features", "_add_to_que",
                    "params_dict"}:
            if delete_key in params_dict.keys():
                del params_dict[delete_key]


        for k,v in {k:v for k,v in params_dict.items()}.items():
            if k not in parameters:
                del params_dict[k]
            elif isinstance(v,set):
                params_dict[k] = list(v)

        self.__function_pipe.append((function_name,
                                     params_dict))

        # Generate new json file name with proper file/folder output attributes
        if len(self.__function_pipe) == 1 and not self.__json_file_name:
            FileOutput.__init__(self,
                                f'_Extras/Pipeline Structure/Data Pipeline Segments/{self.__object_type}')
            all_json_files = get_all_files_from_path(self.folder_path,
                                                     ".json")
            while True:
                random_file_name = create_hex_decimal_string().upper()
                if random_file_name not in all_json_files:
                    break

            self.__segment_id = random_file_name
            self.__json_file_name = random_file_name + ".json"

        # Update json file
        if self.__create_file:
            self.__create_json_pipeline_segment_file()

    def __create_json_pipeline_segment_file(self):
        """
        Desc:
            Creates a json file that the segment relates to; this file's
            segment id can be used later used in the init to get all functions
            that object already used.
        """
        json_dict = dict()

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

        # Generate pipeline segment file
        dict_to_json_file(json_dict,
                          self.folder_path,
                          self.__json_file_name)

    def __configure_pipeline_segment_with_existing_file(self):
        """
        Desc:
            Attempts to get a json file and then re_init the 'function_pipe'
            and the 'json_file_name'.
        """

        FileOutput.__init__(self,
                            f'_Extras/Pipeline Structure/Data Pipeline Segments/{self.__object_type}')

        self.__function_pipe = deque()
        self.__json_file_name = copy.deepcopy(self.__segment_id) + ".json"

        # File/Folder error checks
        if not os.path.exists(self.folder_path):
            raise PipelineSegmentError(
                "Couldn't find the pipeline segment's folder when trying to configure this object with the provided json file.")
        if not os.path.exists(self.folder_path + copy.deepcopy(self.__json_file_name)):
            raise PipelineSegmentError(
                f"Couldn't find the pipeline segment's file named '{self.__json_file_name}' in the pipeline's directory when trying to configure this object with the provided json file.")

        json_dict = json_file_to_dict(
            self.folder_path + copy.deepcopy(self.__json_file_name))

        # Push functions into function pipe
        for function_order in range(1,json_dict["Pipeline Segment"]["Function Count"] + 1):
            function_name = list(json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function Order {function_order}"].keys())[0]
            params_dict = json_dict["Pipeline Segment"]["Functions Performed Order"][f"Function Order {function_order}"][function_name]["Params Dict"]
            self.__function_pipe.append((function_name,
                                         params_dict))

