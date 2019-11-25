from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow._hidden.custom_exceptions import SnapshotMismatchError
from eflow.utils.pandas_utils import df_to_image
from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import convert_to_filename

import copy
from IPython.display import display
from matplotlib import pyplot as plt
import warnings

class DataAnalysis(FileOutput):

    def __init__(self,
                 df_features,
                 project_name,
                 overwrite_full_path=None):

        self.__df_features = copy.deepcopy(df_features)

        FileOutput.__init__(self,
                            project_name,
                            overwrite_full_path)



    def save_plot(self,
                  df,
                  filename="Unknown filename",
                  sub_dir=None,
                  dataframe_snapshot=True,
                  suppress_runtime_errors=True,
                  compare_shape=True,
                  compare_feature_names=True,
                  compare_random_values=True):
        """
        Desc:
            Checks the passed data to see if a plot can be saved; raises
            an error if it can't.

        Args:
            df:
                Pandas DataFrame object

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

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
                                                  self.__df_features,
                                                  directory_path=self.folder_path,
                                                  sub_dir=f"{sub_dir}/_Extras")
            # Create the png to save
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

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
                           filename="Unknown filename",
                           sub_dir=None,
                           dataframe_snapshot=True,
                           suppress_runtime_errors=True,
                           compare_shape=True,
                           compare_feature_names=True,
                           compare_random_values=True,
                           table=None):
        """
        Desc:
            Checks the passed data to see if a table can be saved as a plot;
            raises an error if it can't.

        Args:
            df:
                Pandas DataFrame object

            feature_name:
                Specified feature column name.

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.
        """

        try:
            if not isinstance(sub_dir, str):
                raise TypeError(
                    f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            if dataframe_snapshot:
                df_snapshot = DataFrameSnapshot(compare_shape=compare_shape,
                                                compare_feature_names=compare_feature_names,
                                                compare_random_values=compare_random_values)
                df_snapshot.check_create_snapshot(df,
                                                  self.__df_features,
                                                  directory_path=self.folder_path,
                                                  sub_dir=f"{sub_dir}/_Extras")

            # Convert value counts dataframe to an image
            df_to_image(table,
                        self.folder_path,
                        sub_dir,
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)

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