from eFlow._Hidden.Objects.enum import *
from eFlow._Hidden.Constants import *
import os
import copy

class FileOutput:

    def __init__(self,
                 project_name,
                 overwrite_full_path=None):
        # Setup project structure
        if not overwrite_full_path:
            parent_structure = "/" + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME \
                               + "/" + project_name + "/"
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=
                                  os.getcwd() + parent_structure)
        else:
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=overwrite_full_path)

    def get_output_folder(self):
        return copy.deepcopy(self.__PROJECT.PATH_TO_OUTPUT_FOLDER)
