from eflow._hidden.constants import GRAPH_DEFAULTS
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import convert_to_filename, correct_directory_path
from eflow.utils.pandas_utils import data_types_table, missing_values_table, df_to_image
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow._hidden.parent_objects import FileOutput
import copy
from IPython.display import display

import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
import warnings


__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class NullAnalysis(FileOutput):
    """
        Analyzes a pandas dataframe's object for null data; creates visuals
        like graphs and tables.
    """

    def __init__(self,
                 df_features,
                 project_sub_dir="",
                 project_name="Missing Data",
                 overwrite_full_path=None,
                 notebook_mode=True):
        """
        Args:
            project_sub_dir:
                Appends to the absolute directory of the output folder

            project_name:
                Creates a parent or "project" folder in which all sub-directories
                will be inner nested.

            overwrite_full_path:
                Overwrites the path to the parent folder.

            notebook_mode:
                If in a python notebook display visualizations in the notebook.
        """

        FileOutput.__init__(self,
                            f'{project_sub_dir}/{project_name}',
                            overwrite_full_path)

        self.__df_features = copy.deepcopy(df_features)
        self.__notebook_mode = copy.deepcopy(notebook_mode)

        # Determines if the perform was called to see if we need to re-check
        # the dataframe.
        self.__called_from_perform = False


    def perform_analysis(self,
                         df,
                         dataset_name,
                         display_visuals=True,
                         save_file=True,
                         dataframe_snapshot=True,
                         null_features_only=False):
        """
        Desc:
            Perform all public methods of the NullAnalysis object.

        Args:
            df:
                Pandas Dataframe object.

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            save_file:
                Boolean value to whether or not to save the file.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            null_features_only:
                Dataframe will pass on null features for the visualizations
        """
        try:
            self.__called_from_perform = False

            if df is not None:

                # All functionality is meaningless without getting past the
                # following check; exit function
                if not self.__check_dataframe(df):
                    print("Exiting perform data_analysis function call")
                    return None

                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")


                # Set to true to represent the function call was made with perform
                self.__called_from_perform = True

                self.data_types_table(df,
                                      dataset_name,
                                      display_visuals=display_visuals,
                                      save_file=save_file)
                print("\n\n")
                # --------------------------------------
                self.missing_values_table(df,
                                          dataset_name)
                print("\n\n")
                # --------------------------------------
                self.plot_null_bar_graph(df,
                                         dataset_name,
                                         null_features_only=null_features_only,
                                         display_visuals=display_visuals,
                                         save_file=save_file)
                print("\n\n")
                # --------------------------------------
                self.plot_null_matrix_graph(df,
                                            dataset_name,
                                            null_features_only=null_features_only,
                                            display_visuals=display_visuals,
                                            save_file=save_file)
                print("\n\n")
                # --------------------------------------
                self.plot_null_heatmap_graph(df,
                                             dataset_name,
                                             display_visuals=display_visuals,
                                             save_file=save_file)
                print("\n\n")
                # --------------------------------------
                self.plot_null_dendrogram_graph(df,
                                                dataset_name,
                                                null_features_only=null_features_only,
                                                display_visuals=display_visuals,
                                                save_file=save_file)
                print("\n\n")

        finally:
            self.__called_from_perform = False


    def plot_null_matrix_graph(self,
                               df,
                               dataset_name,
                               display_visuals=True,
                               filename=None,
                               save_file=True,
                               dataframe_snapshot=True,
                               null_features_only=False,
                               filter=None,
                               n=0,
                               p=0,
                               sort=None,
                               figsize=GRAPH_DEFAULTS.NULL_FIGSIZE,
                               width_ratios=(15, 1),
                               color=(.027, .184, .373),
                               fontsize=16,
                               labels=None,
                               sparkline=True,
                               inline=False,
                               freq=None):
        """
        Desc (Taken from missingno):
            A matrix visualization of the nullity of the given DataFrame then
            pushes the image to output folder.

        Args:
            df:
                Pandas dataframe object

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            save_file:
                Boolean value to whether or not to save the file.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            null_features_only:
                Dataframe will pass on null features for the visualizations

            Please read the offical documentation at for more about the parameters:
            Link: https://github.com/ResidentMario/missingno

            Note:
                Changed the default color of the bar graph because I thought it
                was ugly.
        """
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno

        if not self.__called_from_perform:
            if not self.__check_dataframe(df):
                print("Null matrix couldn't be generated because there is "
                      "no missing data to display!")
                return None

        null_sorted_features, null_features = self.__sort_features_by_nulls(df)

        if null_features_only:
            selected_features = null_features
        else:
            selected_features = null_sorted_features

        if display_visuals:
            print("Generating graph for null matrix graph...")

        plt.close()
        msno.matrix(df[selected_features],
                    filter=filter,
                    n=n,
                    p=p,
                    sort=sort,
                    figsize=figsize,
                    width_ratios=width_ratios,
                    color=color,
                    fontsize=fontsize,
                    labels=labels,
                    sparkline=sparkline,
                    inline=inline,
                    freq=freq)

        if not filename:
            filename = "Missing data matrix graph"


        if save_file:

            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))

        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()


    def plot_null_bar_graph(self,
                            df,
                            dataset_name,
                            display_visuals=True,
                            filename=None,
                            save_file=True,
                            dataframe_snapshot=True,
                            null_features_only=False,
                            figsize=GRAPH_DEFAULTS.NULL_FIGSIZE,
                            fontsize=16,
                            labels=None,
                            log=False,
                            color=GRAPH_DEFAULTS.NULL_COLOR,
                            inline=False,
                            filter=False,
                            n=0,
                            p=0,
                            sort=None):
        """
        Desc (Taken from missingno):
            A bar graph visualization of the nullity of the given DataFrame then
            pushes the image to output folder.

        Args:
            df:
                Pandas dataframe object

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            null_features_only:
                Dataframe will pass on null features for the visualizations

            display_visuals:
                Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file:
                Boolean value to whether or not to save the file.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            Please read the offical documentation for more about the parameters:
            Link: https://github.com/ResidentMario/missingno

            Note:
                Changed the default color of the bar graph because I thought it
                was ugly.
        """
        # Credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno

        if not self.__called_from_perform:
            if not self.__check_dataframe(df):
                print("Null bar graph couldn't be generated because there is "
                      "no missing data to display!")
                return None

        null_sorted_features, null_features = self.__sort_features_by_nulls(df)

        if null_features_only:
            selected_features = null_features
        else:
            selected_features = null_sorted_features

        print("Generating graph for null bar graph...")

        plt.close()
        ax = msno.bar(df[selected_features],
                      figsize=figsize,
                      log=log,
                      fontsize=fontsize,
                      labels=labels,
                      color=color,
                      inline=inline,
                      filter=filter,
                      n=n,
                      p=p,
                      sort=sort)

        # Annotation
        props = dict(boxstyle='round',
                     facecolor="#FFFFFF",
                     alpha=0)
        ax.text(0.05,
                1.13,
                f"Clean data is {df.shape[0]} entries",
                transform=ax.transAxes,
                fontsize=10,
                size=17,
                verticalalignment='top',
                bbox=props)

        # Sets filename with a default name
        if not filename:
            filename = "Missing data bar graph"

        if save_file:
            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            # Convert plot to png
            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))

        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()

    def plot_null_heatmap_graph(self,
                                df,
                                dataset_name,
                                display_visuals=True,
                                filename=None,
                                save_file=True,
                                dataframe_snapshot=True,
                                inline=False,
                                filter=None,
                                n=0,
                                p=0,
                                sort=None,
                                figsize=GRAPH_DEFAULTS.NULL_FIGSIZE,
                                fontsize=16,
                                labels=True,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                cbar=True):
        """
        Desc (Taken from missingno):
            Presents a `seaborn` heatmap visualization of nullity correlation
            in the given DataFrame.

        Args:
            df:
                Pandas dataframe object

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file:
                Boolean value to whether or not to save the file.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            Please read the offical documentation for more about the parameters:
            Link: https://github.com/ResidentMario/missingno

            Note:
                Changed the default color of the bar graph because I thought it
                was ugly.
        """
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno

        if not self.__called_from_perform:
            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__check_dataframe(df):
                print("Null heatmap graph couldn't be generated because there"
                      "is no missing data to display!")
                return None

        print("Generating graph for null heatmap...")

        # -----
        plt.close()
        msno.heatmap(df,
                     inline=inline,
                     filter=filter,
                     n=n,
                     p=p,
                     sort=sort,
                     figsize=figsize,
                     fontsize=fontsize,
                     labels=labels,
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     cbar=cbar)

        # Sets filename with a default name
        if not filename:
            filename = "Missing data heatmap graph"

        if save_file:
            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            # Convert plot to png
            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))

        if self.__notebook_mode and display_visuals:
            plt.show()
        plt.close()


    def plot_null_dendrogram_graph(self,
                                   df,
                                   dataset_name,
                                   display_visuals=True,
                                   filename=None,
                                   save_file=True,
                                   dataframe_snapshot=True,
                                   null_features_only=False,
                                   method='average',
                                   filter=None,
                                   n=0,
                                   p=0,
                                   orientation=None,
                                   figsize=GRAPH_DEFAULTS.NULL_FIGSIZE,
                                   fontsize=16,
                                   inline=False):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        Desc (Taken from missingno):
            Fits a `scipy` hierarchical clustering algorithm to the given
            DataFrame's variables and visualizes the results as
            a `scipy` dendrogram.

        Args:
            df:
                Pandas dataframe object

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file:
                Boolean value to whether or not to save the file.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            null_features_only:
                Dataframe will pass on null features for the visualizations

            Please read the offical documentation for more about the parameters:
            Link: https://github.com/ResidentMario/missingno
        """

        if not self.__called_from_perform:
            if not self.__check_dataframe(df):
                print("Null dendrogram graph couldn't be generated because"
                      " there is no missing data to display!")
                return None

        null_sorted_features, null_features = self.__sort_features_by_nulls(df)

        if null_features_only:
            selected_features = null_features
        else:
            selected_features = null_sorted_features

        print("Generating graph for null dendrogram graph...")

        plt.close()
        msno.dendrogram(df[selected_features],
                        method=method,
                        filter=filter,
                        n=n,
                        p=p,
                        orientation=orientation,
                        figsize=figsize,
                        fontsize=fontsize,
                        inline=inline)

        # Sets filename with a default name
        if not filename:
            filename = f"Missing data dendrogram graph {method}"

        if save_file:

            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            # Convert plot to png
            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))


        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()

    def missing_values_table(self,
                             df,
                             dataset_name,
                             display_visuals=True,
                             filename=None,
                             save_file=True,
                             dataframe_snapshot=True):
        """
        Desc:
            Creates/Saves a Pandas DataFrame object giving the percentage of
            the null data for the original DataFrame columns.

        Args:
            df:
                Pandas DataFrame object

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                        Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file:
                Boolean value to whether or not to save the file.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.
        """

        if not self.__called_from_perform:
            if not self.__check_dataframe(df):
                print("Couldn't create missing values table because"
                      " there is no missing data to display!")
                return None

        print("Creating missing values table...")

        if not self.__called_from_perform:
            self.__check_dataframe(df)

        mis_val_table_ren_columns = missing_values_table(df)

        print(f"Your selected dataframe has {str(df.shape[1])} columns.\n"
              f"That has {str(mis_val_table_ren_columns.shape[0])} columns missing data.\n")

        if self.__notebook_mode:
            if display_visuals:
                display(mis_val_table_ren_columns)
        else:
            if display_visuals:
                print(mis_val_table_ren_columns)

        # Sets filename with a default name
        if not filename:
            filename = "Missing Data Table"

        # ---
        if save_file:

            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            plt.close()
            df_to_image(mis_val_table_ren_columns,
                        self.folder_path,
                        f"{dataset_name}/Tables",
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)

    def data_types_table(self,
                         df,
                         dataset_name,
                         display_visuals=True,
                         filename=None,
                         save_file=True,
                         dataframe_snapshot=True):
        """
        Desc:
            Creates/Saves a pandas dataframe of features and their found types
            in the dataframe.

        Args:
            df:
                Pandas DataFrame object

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file:
                Saves file if set to True; doesn't if set to False.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.
        """
        if not self.__called_from_perform:
            if not df.shape[0]:
                print("Empty dataframe found! This function requires a dataframe"
                      "in both rows and columns.")
                return None

        print("Creating data types table...")

        dtypes_df = data_types_table(df)

        print(f"Your selected dataframe has {df.shape[1]} features.")

        if self.__notebook_mode:
            if display_visuals:
                display(dtypes_df)
        else:
            if display_visuals:
                print(dtypes_df)

        # Sets filename with a default name
        if not filename:
            filename = "Data Types Table"

        if save_file:
            # Compares the json file snapshot to passed dataframe's snapshot
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")
            plt.close()
            df_to_image(dtypes_df,
                        self.folder_path,
                        f"{dataset_name}/Tables",
                        convert_to_filename(filename),
                        show_index=True)

    def __check_dataframe(self,
                          df):
        """
        Args:
            df:
                Pandas Dataframe object.

        Returns:
            Returns backs a bool to determine whether or not to null analysis
            method should work with it.

        Note:
            I only made this function in case I needed to do more error checks
            in the future.
        """

        passed_check = True

        if not df.isnull().values.any() or df.shape[0] == 0:
            passed_check = False

        return passed_check

    def __sort_features_by_nulls(self,
                                 df):
        """
        Desc:
            Sorts a dataframe by data containing the most nulls to least nulls.

        Args:
            df:
                Pandas Dataframe object.

        Returns:
            Returns back the sorted order of features and the features that
            contain null.
        """

        # Perform sort of nulls
        features = df.isnull().sum().index.tolist()
        null_values = df.isnull().sum().values.tolist()
        null_values, null_sorted_features = zip(*sorted(zip(null_values,
                                                            features)))

        # Get list and reverse sequence
        null_values = list(null_values)
        null_sorted_features = list(null_sorted_features)

        null_sorted_features.reverse()
        null_values.reverse()
        # -------------------------------------

        # Iterate until through feature values until no nulls feature is found
        for feature_index, value in enumerate(null_values):
            if value == 0:
                break

        null_features = null_sorted_features[0:feature_index]

        return null_sorted_features, null_features