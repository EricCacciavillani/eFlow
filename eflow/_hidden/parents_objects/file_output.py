from eflow._hidden.constants import *
from eflow.utils.sys_utils import check_create_dir_structure
from eflow.utils.string_utils import correct_directory_path

import os
import copy


class FileOutput(object):

    def __init__(self,
                 project_name,
                 overwrite_full_path=None):
        # Setup project structure
        if not overwrite_full_path:
            parent_structure = "/" + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME \
                               + "/" + project_name + "/"

            check_create_dir_structure(os.getcwd(),
                                       parent_structure)

            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=
                                  correct_directory_path(
                                      os.getcwd() + parent_structure))

        # This path must already exist
        else:
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=correct_directory_path(overwrite_full_path))

    def get_output_folder(self):
        return copy.deepcopy(self.__PROJECT.PATH_TO_OUTPUT_FOLDER)
