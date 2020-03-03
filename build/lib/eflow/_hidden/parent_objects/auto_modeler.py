from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow._hidden.custom_exceptions import SnapshotMismatchError
from eflow.utils.pandas_utils import df_to_image, generate_meta_data
from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import convert_to_filename

import copy
from IPython.display import display
from matplotlib import pyplot as plt
import warnings

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class AutoModeler(FileOutput):
    """
        All objects in data_analysis folder of eflow are related to this object.
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
                  sub_dir="",
                  filename="Unknown filename",
                  suppress_runtime_errors=True):
        """
        Desc:
            Checks the passed data to see if a plot can be saved; raises
            an error if it can't.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.
        """
        try:

            if not isinstance(sub_dir, str):
                raise TypeError(
                    f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Create the png to save
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        # Always raise snapshot error
        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close("all")
            if suppress_runtime_errors:
                warnings.warn(
                    f"An error occured when trying to save the plot:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def save_table_as_plot(self,
                           df,
                           sub_dir="",
                           filename="Unknown filename",
                           suppress_runtime_errors=True,
                           show_index=False,
                           format_float_pos=2):
        """
        Desc:
            Checks the passed data to see if a table can be saved as a plot;
            raises an error if it can't.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object of all data.

            sub_dir:
                Directory to generate on top of the already made directory (self.folder_path).

            overwrite_full_path:
                A pre defined directory that already exists.

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            table: pd.Dataframe
                Dataframe object to convert to plot.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            meta_data:bool
                If set to true then it will generate meta data on the dataframe.

            show_index: bool
                Show index of the saved dataframe.
        """

        try:
            if not isinstance(sub_dir, str):
                raise TypeError(
                    f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Convert dataframe to plot
            df_to_image(df,
                        self.folder_path,
                        sub_dir,
                        convert_to_filename(filename),
                        show_index=show_index,
                        format_float_pos=format_float_pos)

        # Always raise snapshot error
        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close("all")
            if suppress_runtime_errors:
                warnings.warn(
                    f"An error occured when trying to save the table as a plot:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e