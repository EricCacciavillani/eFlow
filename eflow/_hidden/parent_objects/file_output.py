from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow._hidden.constants import SYS_CONSTANTS
from eflow.utils.sys_utils import create_dir_structure
from eflow.utils.string_utils import correct_directory_path
import os
import copy

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class FileOutput(object):
    """
        Ensures a folder is part of eflow's main output stream and creates a
        sub directory based on the arg project_name.
        Note:
              If 'project_name' string is formatted in a relative path form
              then it will generate the provided directory.

              Ex:
                Dir A/Best Project. Will generate folders "Dir A" and
                Best Project.
    """

    def __init__(self,
                 project_name,
                 overwrite_full_path=None):
        """
        Args:
            project_name: string
                Sub directory to create on top of the directory
                'PARENT_OUTPUT_FOLDER_NAME'.

            overwrite_full_path: string
                The passed directory path must already exist. Will completely
                ignore the project name and attempt to point to this already
                created directory.
        """

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

            # Path doesn't contain eflow's main output
            if f"/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/" not in overwrite_full_path:
                raise UnsatisfiedRequirments(f"Directory path must have {SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME} "
                                             f"as a directory name or this program will not work correctly.")

            # Unknown path found
            if not os.path.exists(overwrite_full_path):
                raise SystemError("The path must already be defined in full on "
                                  "your system to use a different directory "
                                  "structure than orginally intended.")

            tmp_path = overwrite_full_path

        from eflow._hidden.general_objects import enum
        self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=tmp_path,
                              RELATIVE_PATH_TO_OUTPUT_FOLDER=tmp_path.split(f"/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/")[1])

    @property
    def folder_path(self):
        """
        Desc:
            Path to folder
        """
        return copy.deepcopy(self.__PROJECT.PATH_TO_OUTPUT_FOLDER)

    @property
    def relative_folder_path(self):
        """
        Desc:
            Relative path to folder.

        Note:
            Ignoring the parent output folder.
        """
        return copy.deepcopy(self.__PROJECT.RELATIVE_PATH_TO_OUTPUT_FOLDER)
