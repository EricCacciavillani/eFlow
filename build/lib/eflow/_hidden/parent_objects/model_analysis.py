from eflow._hidden.parent_objects import FileOutput
from eflow.utils.pandas_utils import df_to_image
from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import convert_to_filename
from eflow.utils.sys_utils import create_dir_structure,correct_directory_path

import copy
from IPython.display import display
from matplotlib import pyplot as plt
import warnings
import numpy as np
import pandas as pd

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class ModelAnalysis(FileOutput):
    """
        All objects in model_analysis folder of eflow are related to this object.
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
        # Create/Setup project directory
        FileOutput.__init__(self,
                            project_name,
                            overwrite_full_path)

    def save_plot(self,
                  filename="Unknown filename",
                  sub_dir=""):
        """
        Desc:
            Checks the passed data to see if a plot can be saved; raises
            an error if it can't.

        Args:

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

        """

        if not isinstance(sub_dir, str):
            raise TypeError(
                f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

        # Create the png to save
        create_plt_png(self.folder_path,
                       sub_dir,
                       convert_to_filename(filename))


    def save_table_as_plot(self,
                           table,
                           filename="Unknown filename",
                           sub_dir="",
                           show_index=False):
        """
        Desc:
            Checks the passed data to see if a table can be saved as a plot;
            raises an error if it can't.

        Args:

            table: pd.Dataframe
                Dataframe object to convert to plot.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            show_index: bool
                Show index of the saved dataframe.
        """

        if not isinstance(sub_dir, str):
            raise TypeError(
                f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")


        # Convert dataframe to plot
        df_to_image(table,
                    self.folder_path,
                    sub_dir,
                    convert_to_filename(filename),
                    show_index=show_index,
                    format_float_pos=2)

    def generate_matrix_meta_data(self,
                                  X,
                                  sub_dir):
        """
        Desc:
            Generates files/graphics in the proper directory for the matrix.

        Args:
            X: list of list; numpy array of numpy array or numpy matrix
                Numpy matrix

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.
        """

        # Convert to numpy array if possible
        X = np.array(X)

        create_dir_structure(self.folder_path,
                             correct_directory_path(sub_dir + "/Meta Data"))

        output_folder_path = correct_directory_path(self.folder_path)

        # Create files relating to dataframe's shape
        shape_df = pd.DataFrame.from_dict({'Rows': [X.shape[0]],
                                           'Columns': [X.shape[1]]})
        df_to_image(shape_df,
                    f"{output_folder_path}/{sub_dir}",
                    "Meta Data",
                    "Matrix Shape Table",
                    show_index=False)