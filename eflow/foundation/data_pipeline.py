from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.parent_objects import DataPipelineSegment
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, PipelineError
from eflow.utils.string_utils import create_hex_decimal_string,correct_directory_path
from eflow.utils.sys_utils import create_json_file_from_dict, get_all_files_from_path, check_create_dir_structure, json_file_to_dict
# from eflow.data_pipeline_segments import DataCleaner
from eflow.data_pipeline_segments import DataTransformer
from eflow._hidden.constants import SYS_CONSTANTS
import copy
import os
from collections import deque

class DataPipeline(FileOutput):
    """
        Houses all pipeline segments. Able to load these segments in with
        a pipeline json file and/or modify id. You can add more pipeline
        segments if needed which will modify the pipeline's json file. Allows
        a pandas dataframe to be processed by 'perform_pipeline'.
    """
    def __init__(self,
                 pipeline_name,
                 pipeline_modify_id=None):
        """
        pipeline_name (str):
            Points to/generates a folder based on the pipeline's name.

        pipeline_modify_id (str,NoneType):
            If set to 'None' then will point the 'root' or the main template
            of the pipeline.
        """
        #
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

        # Json file does exist; init DataPipeline object correctly
        if configure_existing_file:
            print("Now configuring object with proper pipeline segments...")
            self.__configure_pipeline_with_existing_file()

    @property
    def file_path(self):
        """
        Returns/Desc:
            File path with file name.
        """
        if len(self.__pipeline_segment_deque) == 0:
            raise PipelineError(
                "The pipeline has not added any segments yet."
                " Please add segments to this object.")
        else:
            return self.folder_path + copy.deepcopy(self.__json_file_name)

    @property
    def file_name(self):
        """
        Returns/Desc:
            File name with extension.
        """
        if len(self.__pipeline_segment_deque) == 0:
            raise PipelineError(
                "The pipeline has not added any segments yet."
                " Please add segments to this object.")
        else:
            return copy.deepcopy(self.__json_file_name)


    def add(self,
            segment_name,
            pipeline_segment_obj):
        """
        segment_name (str):
            A aliased name to refer to this segment.

        pipeline_segment_obj (child of DataPipelineSegment):
            A child object of type DataPipelineSegment.

        Returns/Desc:
            Attempts to add a pipeline segment object to the objects que and
            update it's related json object.
        """

        # Type check
        if not isinstance(pipeline_segment_obj,
                          DataPipelineSegment):

            raise UnsatisfiedRequirments(f"Expected a 'DataPipelineSegment' object; received '{type(pipeline_segment_obj)}'")

        # Check if alias has already been used
        if segment_name in self.__pipeline_segment_names:
            raise PipelineError(f"The '{segment_name}' pipeline segment is already in this pipeline. Please choose a different segment name.")

        # Check if the pipeline segment has already been used
        segment_path_id = pipeline_segment_obj.relative_folder_path + pipeline_segment_obj.file_name
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
            # Que has yet to have data pushed; set up output directory
            if len(self.__pipeline_segment_deque) == 0:
                FileOutput.__init__(self,
                                    f'_Extras/JSON Files/Data Pipeline/{self.__pipeline_name}')


        # Update data types for error checking
        self.__pipeline_segment_names.add(segment_name)
        self.__pipeline_segment_path_id.add(segment_path_id)

        # Main component of the project
        self.__pipeline_segment_deque.append((segment_name,
                                              segment_path_id,
                                              pipeline_segment_obj))

        # Update/Create the json file
        self.__create_json_pipeline_file()


    def __create_json_pipeline_file(self):
        """
        Returns/Desc:
            Creates a dict based on the given contents of the variable
            'self.__pipeline_segment_deque' to convert to a json file.
            This file will later be used to instruct our object to execute
            specific code.
        """

        # -------------
        json_dict = dict()
        segment_order = 1

        json_dict["Pipeline Name"] = self.__pipeline_name
        json_dict["Pipeline Segment Order"] = dict()
        for segment_name, segment_path_id, pipeline_segment_obj in self.__pipeline_segment_deque:
            json_dict["Pipeline Segment Order"][segment_order] = dict()
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Path"] = segment_path_id
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Type"] = pipeline_segment_obj.__class__.__name__
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment Name"] = segment_name
            json_dict["Pipeline Segment Order"][segment_order]["Pipeline Segment ID"] = segment_path_id.split("/")[-1].split(".")[0]

            segment_order += 1

        json_dict["Pipeline Segment Count"] = segment_order - 1

        # Create a folder for all non-root json files.
        if self.__pipeline_modify_id:
            check_create_dir_structure(self.folder_path,
                                       "/Modified Pipelines")
            create_json_file_from_dict(json_dict,
                                       self.folder_path + "/Modified Pipelines",
                                       self.__json_file_name)
        # Root json files only
        else:
            create_json_file_from_dict(json_dict,
                                       self.folder_path,
                                       self.__json_file_name)

    def __configure_pipeline_with_existing_file(self):
        """
        Returns/Desc:
            Changes the objects variables based on the provided json file.
        """

        # Error check paths
        if not os.path.exists(self.folder_path):
            raise PipelineError("Couldn't find the pipeline's folder when trying to configure this object with the provided json file.")

        # ------
        if not os.path.exists(self.folder_path + copy.deepcopy(self.__json_file_name)):
            raise PipelineError(f"Couldn't find the pipeline's file named '{self.file_name}' in the pipeline's directory when trying to configure this object with the provided json file.")

        # Reset variables
        self.__pipeline_segment_deque = deque()
        self.__pipeline_segment_names = set()
        self.__pipeline_segment_path_id = set()

        json_dict = json_file_to_dict(self.folder_path + copy.deepcopy(self.__json_file_name))

        # Iterate through dict to init variables properly
        for segment_order in range(1,json_dict["Pipeline Segment Count"] + 1):
            segment_type = json_dict["Pipeline Segment Order"][str(segment_order)]["Pipeline Segment Type"]
            segment_name = \
            json_dict["Pipeline Segment Order"][str(segment_order)][
                "Pipeline Segment Name"]
            segment_path_id = \
            json_dict["Pipeline Segment Order"][str(segment_order)][
                "Pipeline Segment Path"]
            segment_id = json_dict["Pipeline Segment Order"][str(segment_order)]["Pipeline Segment ID"]

            pipeline_segment_obj = None

            pipeline_segment_obj = eval(f"{segment_type}(\"{segment_id}\")\n")

            if not pipeline_segment_obj:
                raise PipelineError(f"An unknown error has occurred with finding the correct pipeline segment for '{segment_type}' segment!")

            self.__pipeline_segment_names.add(segment_name)
            self.__pipeline_segment_path_id.add(segment_path_id)

            self.__pipeline_segment_deque.append((segment_name,
                                                  segment_path_id,
                                                  pipeline_segment_obj))

    def perform_pipeline(self,
                         df):
        """
        df:
            Pandas Dataframe object to be transformed by the pipeline.

        Returns/Desc:
            Applies a Pandas Dataframe object to all functions on all segments
            in the pipeline.
        """
        for _, _, pipeline_segment in self.__pipeline_segment_deque:
            pipeline_segment.perform_segment(df)

    def generate_code(self,
                      generate_file=True,
                      add_libs=True):
        """
        generate_file (bool):
            Determines whether or not to generate a python file 
        add_libs:

        Returns/Desc:
            Applies
        """
        if len(self.__pipeline_segment_deque) == 0:
            raise PipelineError("Pipeline needs a segment to generate code.")

        generated_code = []

        if add_libs:
            libs_code = list()
            libs_code.append("from eflow.utils.math_utils import *")
            libs_code.append("from eflow.utils.image_utils import *")
            libs_code.append("from eflow.utils.pandas_utils import *")
            libs_code.append("from eflow.utils.modeling_utils import *")
            libs_code.append("from eflow.utils.string_utils import *")
            libs_code.append("from eflow.utils.misc_utils import *")
            libs_code.append("from eflow.utils.sys_utils import *")
            libs_code.append("")
            generated_code.append(libs_code)


        for segment_name, segment_path_id, pipeline_segment in self.__pipeline_segment_deque:
            tmp_list = pipeline_segment.generate_code(generate_file=False,
                                                      add_libs=False)
            tmp_list.insert(0, f'# Segment Name:    {segment_name}')
            tmp_list.insert(0, f'# Segment Path ID: {segment_path_id}')

            generated_code.append(tmp_list)

        check_create_dir_structure(self.folder_path,
                                   "Generated code")

        if generate_file:
            python_file_name = self.__json_file_name.split(".")[0] + ".py"
            with open(self.folder_path +
                      f'Generated code/{python_file_name}',
                      'w') as filehandle:
                for segment_code in generated_code:
                    for listitem in segment_code:
                        filehandle.write('%s\n' % listitem)
            print(
                f"Generated a python file named/at: {self.folder_path}Generated code/{python_file_name}")
        else:
            return generated_code
