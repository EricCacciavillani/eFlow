from eflow._hidden.objects import FileOutput
import copy
from IPython.display import display

import pandas as pd
import missingno as msno
from matplotlib import pyplot as plt
import warnings
from eflow.utils.image_utils import df_to_image
from eflow._hidden.custom_warnings import DataFrameWarning
from eflow._hidden.constants import GRAPH_DEFAULTS

from eflow.utils.image_utils import create_plt_png
from eflow.utils.string_utils import convert_to_filename, correct_directory_path
from eflow.utils.pandas_utils import data_types_table, missing_values_table
from eflow._hidden.objects import DataFrameSnapshot

class MissingDataAnalysis(FileOutput):
    """
        Analyzes a pandas dataframe's object for null data; creates visuals
        like graphs and tables.
    """

    def __init__(self,
                 sub_dir="",
                 project_name="Missing Data",
                 overwrite_full_path=None,
                 notebook_mode=True):
        """
        sub_dir:
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
                            f'{sub_dir}/{project_name}',
                            overwrite_full_path)
        self.__notebook_mode = copy.deepcopy(notebook_mode)
        self.__called_from_peform = False

    def __check_dataframe(self,
                          df):
        """
        df:
            Pandas Dataframe object.

        Returns/Desc:
            Checks if the dataframe has nulls to analyze.
        """

        passed_check = True

        if not df.isnull().values.any():

            warnings.warn('The given object requires null data to visualize',
                          DataFrameWarning,
                          stacklevel=1000)
            passed_check = False

        if df.shape[0] == 0:
            warnings.warn('Empty Dataframe found',
                          DataFrameWarning,
                          stacklevel=1000)
            passed_check = False

        if not passed_check:
            print(
                "All functionality belonging to this object requires null data!")

        return passed_check

    def __sort_features_by_nulls(self,
                                 df):
        """

        df:
            Pandas Dataframe object.

        Returns/Desc:
            Sorts a dataframe by data containing the most nulls to least nulls.
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

    def perform_analysis(self,
                         df,
                         dataset_name,
                         null_features_only=False,
                         display_visuals=True,
                         save_file=True,
                         dataframe_snapshot=True):
        """
        df:
            Pandas Dataframe object.

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        null_features_only:
            Dataframe will pass on null features for the visualizations

        display_visuals:
            Boolean value to whether or not to display visualizations.

        save_file:
            Boolean value to whether or not to save the file.

        df_features:
            DataFrameTypeHolder object. If initalized we can run correct/error
            analysis on the dataframe. Will save object in a pickle file and
            provided columns if initalized and df_features is not initialized.

        Returns/Desc:
            Perform all functionality of the analysis object.
        """
        try:
            self.__called_from_peform = False

            if df is not None:

                # All functionality is meaningless without getting past the
                # following check; exit function
                if not self.__check_dataframe(df):
                    print("Exiting perform analysis function call")
                    return None

                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")


                self.__called_from_peform = True

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
            self.__called_from_peform = False


    def plot_null_matrix_graph(self,
                               df,
                               dataset_name,
                               null_features_only=False,
                               display_visuals=True,
                               filename=None,
                               save_file=True,
                               dataframe_snapshot=True,
                               filter=None,
                               n=0,
                               p=0,
                               sort=None,
                               figsize=(25, 10),
                               width_ratios=(15, 1),
                               color=(.027, .184, .373),
                               fontsize=16,
                               labels=None,
                               sparkline=True,
                               inline=False,
                               freq=None,
                               ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        null_features_only:
            Dataframe will pass on null features for the visualizations

        display_visuals:
            Boolean value to whether or not to display visualizations.

        save_file:
            Boolean value to whether or not to save the file.

        filename:
            If set to 'None' will default to a given filename;
            else will use the passed string

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Desc (Taken from missingno):
            A matrix visualization of the nullity of the given DataFrame.
        """

        if not self.__called_from_peform:
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
                    freq=freq,
                    ax=ax)

        if not filename:
            filename = "Missing data matrix graph"


        if save_file:

            if not self.__called_from_peform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.get_output_folder(),
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))

        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()


    def plot_null_bar_graph(self,
                            df,
                            dataset_name,
                            null_features_only=False,
                            display_visuals=True,
                            filename=None,
                            save_file=True,
                            dataframe_snapshot=True,
                            figsize=(24, 10),
                            fontsize=16,
                            labels=None,
                            log=False,
                            color="#072F5F",
                            inline=False,
                            filter=False,
                            n=0,
                            p=0,
                            sort=None,
                            ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        null_features_only:
            Dataframe will pass on null features for the visualizations

        display_visuals:
            Boolean value to whether or not to display visualizations.

        save_file:
            Boolean value to whether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Desc (Taken from missingno):
            A bar graph visualization of the nullity of the given DataFrame.
        """

        if not self.__called_from_peform:
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
                      sort=sort,
                      ax=ax)

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

        if not filename:
            filename = "Missing data bar graph"

        if save_file:
            if not self.__called_from_peform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.get_output_folder(),
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
                                figsize=(20, 12),
                                fontsize=16,
                                labels=True,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                cbar=True,
                                ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        null_features_only:
            Dataframe will pass on null features for the visualizations

        display_visuals:
            Boolean value to whether or not to display visualizations.

        save_file:
            Boolean value to whether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Desc (Taken from missingno):
            Presents a `seaborn` heatmap visualization of nullity correlation
            in the given DataFrame.
        """

        if not self.__called_from_peform:
            if not self.__check_dataframe(df):
                print("Null heatmap graph couldn't be generated because there"
                      "is no missing data to display!")
                return None

        print("Generating graph for null heatmap...")

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
                     cbar=cbar,
                     ax=ax
                     )

        if not filename:
            filename = "Missing data heatmap graph"

        if save_file:
            if not self.__called_from_peform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.get_output_folder(),
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))

        if self.__notebook_mode and display_visuals:
            plt.show()
        plt.close()


    def plot_null_dendrogram_graph(self,
                                   df,
                                   dataset_name,
                                   null_features_only=False,
                                   display_visuals=True,
                                   filename=None,
                                   save_file=True,
                                   dataframe_snapshot=True,
                                   method='average',
                                   filter=None,
                                   n=0,
                                   p=0,
                                   orientation=None,
                                   figsize=None,
                                   fontsize=16,
                                   inline=False,
                                   ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        null_features_only:
            Dataframe will pass on null features for the visualizations

        display_visuals:
            Boolean value to whether or not to display visualizations.

        save_file:
            Boolean value to whether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Desc (Taken from missingno):
            Fits a `scipy` hierarchical clustering algorithm to the given
            DataFrame's variables and visualizes the results as
            a `scipy` dendrogram.
        """

        if not self.__called_from_peform:
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


        msno.dendrogram(df[selected_features],
                        method=method,
                        filter=filter,
                        n=n,
                        p=p,
                        orientation=orientation,
                        figsize=figsize,
                        fontsize=fontsize,
                        inline=inline,
                        ax=ax)

        if not filename:
            filename = f"Missing data dendrogram graph {method}"

        if save_file:

            if not self.__called_from_peform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.get_output_folder(),
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

        df:
            Pandas DataFrame object

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        save_file:
            Boolean value to whether or not to save the file.

        dataframe_snapshot:
            Boolean value to determine whether or not generate and compare a
            snapshot of the dataframe in the dataset's directory structure.
            Helps ensure that data generated in that directory is correctly
            associated to a dataframe.

        display_visuals:
            Boolean value to whether or not to display visualizations.

        Returns/Descr:
            Creates/Saves a Pandas DataFrame object giving the percentage of
            the null data for the original DataFrame columns.
        """

        if not self.__called_from_peform:
            if not self.__check_dataframe(df):
                print("Couldn't create missing values table because"
                      " there is no missing data to display!")
                return None

        print("Creating missing values table...")

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        mis_val_table_ren_columns = missing_values_table(df)

        print(f"Your selected dataframe has {str(df.shape[1])} columns.\n"
              f"That are {str(mis_val_table_ren_columns.shape[0])} columns.\n")

        if self.__notebook_mode:
            if display_visuals:
                display(mis_val_table_ren_columns)
        else:
            if display_visuals:
                print(mis_val_table_ren_columns)

        if not filename:
            filename = "Missing Data Table"

        # ---
        if save_file:

            if not self.__called_from_peform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")


            df_to_image(mis_val_table_ren_columns,
                        self.get_output_folder(),
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
        df:
            Pandas DataFrame object

        dataset_name:
            The dataset's name; this will create a sub-directory in which your
            generated graph will be inner-nested in.

        display_visuals:
            Boolean value to whether or not to display visualizations.

        save_file:
            Saves file if set to True; doesn't if set to False.

        notebook_mode:
            If in a python notebook display visualizations in the notebook.

        Returns/Desc:
            Creates/Saves a pandas dataframe of features and their found types
            in the dataframe.
        """

        if not df.shape[0]:
            print("Empty dataframe found! This function requires a dataframe"
                  "in both rows and columns.")

        print("Creating data types table...")

        dtypes_df = data_types_table(df)

        print(f"Your selected dataframe has {df.shape[1]} features.")

        if self.__notebook_mode:
            if display_visuals:
                display(dtypes_df)
        else:
            if display_visuals:
                print(dtypes_df)

        if not filename:
            filename = "Data Types Table"

        if save_file:
            if not self.__called_from_peform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.get_output_folder(),
                                                      sub_dir=f"{dataset_name}/_Extras")
            df_to_image(dtypes_df,
                        self.get_output_folder(),
                        f"{dataset_name}/Tables",
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)