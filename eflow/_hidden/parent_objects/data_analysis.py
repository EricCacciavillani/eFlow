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

class DataAnalysis(FileOutput):
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
                  df,
                  df_features,
                  filename="Unknown filename",
                  sub_dir="",
                  dataframe_snapshot=True,
                  suppress_runtime_errors=True,
                  compare_shape=True,
                  compare_feature_names=True,
                  compare_random_values=True,
                  meta_data=True):
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

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            compare_shape: bool
                When comparing and creating the dataframe snapshot of the data's
                shape.

            compare_feature_names: bool
                When comparing and creating the dataframe snapshot of the data's
                column names.

            compare_random_values: bool
                When comparing and creating the dataframe snapshot of the data
                sudo random values.

            meta_data:bool
                If set to true then it will generate meta data on the dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.
        """
        try:

            if not isinstance(sub_dir, str):
                raise TypeError(
                    f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Check if dataframe matches saved snapshot; Creates file if needed
            if dataframe_snapshot:
                df_snapshot = DataFrameSnapshot(compare_shape=compare_shape,
                                                compare_feature_names=compare_feature_names,
                                                compare_random_values=compare_random_values)
                df_snapshot.check_create_snapshot(df,
                                                  df_features,
                                                  directory_path=self.folder_path,
                                                  sub_dir=f"{sub_dir}/_Extras")

            # Create the png to save
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))


            if meta_data:
                generate_meta_data(df,
                                   self.folder_path,
                                   f"{sub_dir}" + "/_Extras")


        # Always raise snapshot error
        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close()
            if suppress_runtime_errors:
                warnings.warn(
                    f"An error occured when trying to save the plot:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def save_table_as_plot(self,
                           df,
                           df_features,
                           table,
                           filename="Unknown filename",
                           sub_dir="",
                           dataframe_snapshot=True,
                           suppress_runtime_errors=True,
                           compare_shape=True,
                           compare_feature_names=True,
                           compare_random_values=True,
                           show_index=False,
                           format_float_pos=2,
                           meta_data=True):
        """
        Desc:
            Checks the passed data to see if a table can be saved as a plot;
            raises an error if it can't.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object of all data.\

            df_features: DataFrameTypes from eflow
                DataFrameTypes object.

            table: pd.Dataframe
                Dataframe object to convert to plot.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            compare_shape: bool
                When comparing and creating the dataframe snapshot of the data's
                shape.

            compare_feature_names: bool
                When comparing and creating the dataframe snapshot of the data's
                column names.

            compare_random_values: bool
                When comparing and creating the dataframe snapshot of the data
                sudo random values.

            meta_data:bool
                If set to true then it will generate meta data on the dataframe.

            show_index: bool
                Show index of the saved dataframe.
        """

        try:
            if not isinstance(sub_dir, str):
                raise TypeError(
                    f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Create new snapshot of given data or compare to saved snapshot
            if dataframe_snapshot:
                df_snapshot = DataFrameSnapshot(compare_shape=compare_shape,
                                                compare_feature_names=compare_feature_names,
                                                compare_random_values=compare_random_values)
                df_snapshot.check_create_snapshot(df,
                                                  df_features,
                                                  directory_path=self.folder_path,
                                                  sub_dir=f"{sub_dir}/_Extras")
            if meta_data:
                generate_meta_data(df,
                                   self.folder_path,
                                   f"{sub_dir}" + "/_Extras")

            # Convert dataframe to plot
            df_to_image(table,
                        self.folder_path,
                        sub_dir,
                        convert_to_filename(filename),
                        show_index=show_index,
                        format_float_pos=format_float_pos)

        # Always raise snapshot error
        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close()
            if suppress_runtime_errors:
                warnings.warn(
                    f"An error occured when trying to save the table as a plot:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e