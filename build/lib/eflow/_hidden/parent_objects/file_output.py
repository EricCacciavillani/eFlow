from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow._hidden.constants import SYS_CONSTANTS
from eflow.utils.sys_utils import create_dir_structure
from eflow.utils.string_utils import correct_directory_path
import os
import copy

def enum(**enums):
    return type('Enum', (), enums)

class FileOutput(object):

    def __init__(self,
                 project_name,
                 overwrite_full_path=None):

        # Setup project structure
        if not overwrite_full_path:
            parent_structure = "/" + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME \
                               + "/" + project_name + "/"

            create_dir_structure(os.getcwd(),
                                       parent_structure)
            tmp_path = correct_directory_path(
                os.getcwd() + parent_structure)

        # Trusting the user that this path must already exist
        else:
            overwrite_full_path = correct_directory_path(overwrite_full_path)
            if f"/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/" not in overwrite_full_path:
                raise UnsatisfiedRequirments(f"Directory path must have {SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME} "
                                             f"as a directory name or this program will not work correctly.")

            if not os.path.exists(overwrite_full_path):
                raise SystemError("The path must already be defined in full on "
                                  "your system to use a different directory "
                                  "structure than orginally intended.")

            tmp_path = overwrite_full_path
        self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=tmp_path,
                              RELATIVE_PATH_TO_OUTPUT_FOLDER=tmp_path.split(f"/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/")[1])

    @property
    def folder_path(self):
        return copy.deepcopy(self.__PROJECT.PATH_TO_OUTPUT_FOLDER)

    @property
    def relative_folder_path(self):
        return copy.deepcopy(self.__PROJECT.RELATIVE_PATH_TO_OUTPUT_FOLDER)
