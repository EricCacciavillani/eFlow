from eflow._hidden.constants import GRAPH_DEFAULTS
from eflow._hidden.custom_exceptions import SnapshotMismatchError
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow._hidden.parent_objects import DataAnalysis
from eflow.data_analysis.feature_analysis import FeatureAnalysis
from eflow.utils.pandas_utils import missing_values_table, generate_meta_data
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


class NullAnalysis(DataAnalysis):
    """
        Analyzes a pandas dataframe's object for null data; creates visuals
        like graphs and tables.
    """

    def __init__(self,
                 df_features,
                 project_sub_dir="",
                 project_name="Missing Data",
                 overwrite_full_path=None,
                 notebook_mode=False):
        """
        Args:
            df_features:
                Data

            project_sub_dir: string
                Appends to the absolute directory of the output folder

            project_name: string
                Creates a parent or "project" folder in which all sub-directories
                will be inner nested.

            overwrite_full_path: string
                Overwrites the path to the parent folder.

            notebook_mode: bool
                If in a python notebook display visualizations in the notebook.
        """

        DataAnalysis.__init__(self,
                              f'{project_sub_dir}/{project_name}',
                              overwrite_full_path)

        self.__df_features = copy.deepcopy(df_features)
        self.__notebook_mode = copy.deepcopy(notebook_mode)

        # Determines if the perform was called to see if we need to re-check
        # the dataframe.
        self.__called_from_perform = False

        self.__feature_analysis = FeatureAnalysis(df_features,
                                                  project_name=project_name,
                                                  project_sub_dir=project_sub_dir,
                                                  notebook_mode=notebook_mode)


    def perform_analysis(self,
                         df,
                         dataset_name,
                         display_visuals=True,
                         save_file=True,
                         dataframe_snapshot=True,
                         suppress_runtime_errors=True,
                         display_print=True,
                         null_features_only=False):
        """
        Desc:
            Perform all public methods of the NullAnalysis object.
            Except for feature_analysis_of_null_data.

        Args:
            df: pd.Dataframe
                Pandas Dataframe object.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            null_features_only: bool
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

                generate_meta_data(df,
                                   self.folder_path,
                                   f"{dataset_name}" + "/_Extras")

                # Set to true to represent the function call was made with perform
                self.__called_from_perform = True

                if display_visuals:
                    print("\n\n")
                # --------------------------------------
                self.missing_values_table(df,
                                          dataset_name,
                                          display_visuals=display_visuals,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          display_print=display_print)

                if display_visuals:
                    print("\n\n")
                # --------------------------------------
                self.plot_null_bar_graph(df,
                                         dataset_name,
                                         null_features_only=null_features_only,
                                         display_visuals=display_visuals,
                                         save_file=save_file,
                                         suppress_runtime_errors=suppress_runtime_errors,
                                         display_print=display_print)
                if display_visuals:
                    print("\n\n")
                # --------------------------------------
                self.plot_null_matrix_graph(df,
                                            dataset_name,
                                            null_features_only=null_features_only,
                                            display_visuals=display_visuals,
                                            save_file=save_file,
                                            suppress_runtime_errors=suppress_runtime_errors,
                                            display_print=display_print)
                if display_visuals:
                    print("\n\n")
                # --------------------------------------
                self.plot_null_heatmap_graph(df,
                                             dataset_name,
                                             display_visuals=display_visuals,
                                             save_file=save_file,
                                             suppress_runtime_errors=suppress_runtime_errors,
                                             display_print=display_print)
                if display_visuals:
                    print("\n\n")
                # --------------------------------------
                self.plot_null_dendrogram_graph(df,
                                                dataset_name,
                                                null_features_only=null_features_only,
                                                display_visuals=display_visuals,
                                                save_file=save_file,
                                                suppress_runtime_errors=suppress_runtime_errors,
                                                display_print=display_print)

        finally:
            self.__called_from_perform = False

    def feature_analysis_of_null_data(self,
                                      df,
                                      dataset_name,
                                      target_features=None,
                                      display_visuals=True,
                                      display_print=True,
                                      save_file=True,
                                      suppress_runtime_errors=True,
                                      aggregate_target_feature=True,
                                      selected_features=None,
                                      extra_tables=True,
                                      statistical_analysis_on_aggregates=True,
                                      nan_features=[]):
        """
        Desc:
            Performs all public methods that generate visualizations/insights
            that feature analysis uses on an aggregation of null data in a
            feature.

        Note:
            Pretty much my personal lazy button for running the entire object
            without specifying any method in particular.

        Args:
            df: pd.Dataframe
                Pandas dataframe object

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            target_features: collection of string or None
                A feature name that both exists in the init df_features
                and the passed dataframe.

                Note
                    If init to 'None' then df_features will try to extract out
                    the target feature.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            save_file: bool
                Boolean value to whether or not to save the file.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            extra_tables: bool
                When handling two types of features if set to true this will
                    generate any extra tables that might be helpful.
                    Note -
                        These graphics may create duplicates if you already applied
                        an aggregation in 'perform_analysis'

            statistical_analysis_on_aggregates: bool
                If set to true then the function 'statistical_analysis_on_aggregates'
                will run; which aggregates the data of the target feature either
                by discrete values or by binning/labeling continuous data.

            aggregate_target_feature: bool
                Aggregate the data of the target feature if the data is
                non-continuous data.

                Note
                    In the future I will have this also working with continuous
                    data.

            selected_features: collection object of features
                Will only focus on these selected feature's and will ignore
                the other given features.

            nan_features: collection of strings
                Features names that must contain nan data to aggregate on.

        Raises:
            If an empty dataframe is passed to this function or if the same
            dataframe is passed to it raise error.
        """
        target_features = set(target_features)

        for nan_feature_name in nan_features:

            new_target_features = copy.deepcopy(target_features)

            if nan_feature_name in new_target_features:
                new_target_features.discard(nan_feature_name)

            # No null data ignore feature
            if df[df[nan_feature_name].isna()].shape[0] == 0:
                print(f"No nan data found for {nan_feature_name}")
                continue

            print(f"Feature Analysis on data where {nan_feature_name} = NaN")

            self.__feature_analysis.perform_analysis(
                df[df[nan_feature_name].isna()].drop(columns=[nan_feature_name]),
                dataset_name=dataset_name + "/Feature Analysis of Null Data/" + nan_feature_name + " = NaN",
                target_features=new_target_features,
                display_visuals=display_visuals,
                display_print=display_print,
                save_file=save_file,
                dataframe_snapshot=False,
                suppress_runtime_errors=False,
                aggregate_target_feature=aggregate_target_feature,
                statistical_analysis_on_aggregates=statistical_analysis_on_aggregates,
                selected_features=selected_features,
                extra_tables=extra_tables)


    def plot_null_matrix_graph(self,
                               df,
                               dataset_name,
                               display_visuals=True,
                               display_print=True,
                               filename=None,
                               sub_dir=None,
                               save_file=True,
                               dataframe_snapshot=True,
                               suppress_runtime_errors=True,
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
            df: pd.Dataframe
                Pandas dataframe object

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            save_file: bool
                Boolean value to whether or not to save the file.

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

            null_features_only: bool
                Dataframe will pass on null features for the visualizations

            Please read the offical documentation at for more about the parameters:
            Link: https://github.com/ResidentMario/missingno

            Note:
                Changed the default color of the bar graph because I thought it
                was ugly.
        """
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        try:
            if not self.__called_from_perform:
                if not self.__check_dataframe(df):

                    if display_print:
                        print("Couldn't create missing values table because"
                              " there is no missing data to display!")
                    return None


            null_sorted_features, null_features = self.__sort_features_by_nulls(df)

            if null_features_only:
                selected_features = null_features
            else:
                selected_features = null_sorted_features

            if display_print:
                print("Generating graph for null matrix graph...")

            plt.close("all")
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

                if not sub_dir:
                    sub_dir = f"{dataset_name}/Graphics"

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close("all")

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            plt.close('all')

            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot null matrix raised an error:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e



    def plot_null_bar_graph(self,
                            df,
                            dataset_name,
                            display_visuals=True,
                            filename=None,
                            sub_dir=None,
                            save_file=True,
                            dataframe_snapshot=True,
                            suppress_runtime_errors=True,
                            display_print=True,
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
            df: pd.Dataframe
                Pandas dataframe object

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            null_features_only: bool
                Dataframe will pass on null features for the visualizations

            Please read the offical documentation for more about the parameters:
            Link - https://github.com/ResidentMario/missingno

            Note -
                Changed the default color of the bar graph because I thought it
                was ugly.
        """
        # Credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        try:
            if not self.__called_from_perform:
                if not self.__check_dataframe(df):
                    if display_print:
                        print("Couldn't create missing values table because"
                              " there is no missing data to display!")
                    return None

            null_sorted_features, null_features = self.__sort_features_by_nulls(df)

            if null_features_only:
                selected_features = null_features
            else:
                selected_features = null_sorted_features

            if display_print:
                print("Generating graph for null bar graph...")

            plt.close("all")
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

                if not sub_dir:
                    sub_dir = f"{dataset_name}/Graphics"

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close("all")

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            plt.close('all')

            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot null bar graph raised an error:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def plot_null_heatmap_graph(self,
                                df,
                                dataset_name,
                                display_visuals=True,
                                filename=None,
                                sub_dir=None,
                                save_file=True,
                                dataframe_snapshot=True,
                                suppress_runtime_errors=True,
                                display_print=True,
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
            df: pd.Dataframe
                Pandas dataframe object

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            Please read the offical documentation for more about the parameters:
            Link: https://github.com/ResidentMario/missingno

            Note:
                Changed the default color of the bar graph because I thought it
                was ugly.
        """
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        try:
            if not self.__called_from_perform:
                # Compares the json file snapshot to passed dataframe's snapshot
                if not self.__check_dataframe(df):
                    if display_print:
                        print("Couldn't create missing values table because"
                              " there is no missing data to display!")
                    return None

            if display_print:
                print("Generating graph for null heatmap...")

            # -----
            plt.close("all")
            ax = msno.heatmap(df,
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

            # bottom, top = ax.get_ylim()
            # ax.set_ylim(bottom + 0.5, top - 0.5)

            # Sets filename with a default name
            if not filename:
                filename = "Missing data heatmap graph"

            if save_file:

                if not sub_dir:
                    sub_dir = f"{dataset_name}/Graphics"

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

            if self.__notebook_mode and display_visuals:
                plt.show()
            plt.close("all")

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            plt.close('all')

            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot null heatmap raised an error:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e


    def plot_null_dendrogram_graph(self,
                                   df,
                                   dataset_name,
                                   display_visuals=True,
                                   filename=None,
                                   sub_dir=None,
                                   save_file=True,
                                   dataframe_snapshot=True,
                                   suppress_runtime_errors=True,
                                   display_print=True,
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

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            null_features_only: bool
                Dataframe will pass on only null features for the visualizations

            Please read the offical documentation for more about the parameters:
            Link: https://github.com/ResidentMario/missingno
        """
        try:
            if not self.__called_from_perform:
                if not self.__check_dataframe(df):
                    if display_print:
                        print("Couldn't create missing values table because"
                              " there is no missing data to display!")
                    return None

            null_sorted_features, null_features = self.__sort_features_by_nulls(df)

            if null_features_only:
                selected_features = null_features
            else:
                selected_features = null_sorted_features

            if display_print:
                print("Generating graph for null dendrogram graph...")

            plt.close("all")
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

                if not sub_dir:
                    sub_dir = f"{dataset_name}/Graphics"

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)


            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close("all")

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            plt.close('all')

            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot null dendrogram raised an error:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def missing_values_table(self,
                             df,
                             dataset_name,
                             display_visuals=True,
                             filename=None,
                             sub_dir=None,
                             save_file=True,
                             dataframe_snapshot=True,
                             suppress_runtime_errors=True,
                             display_print=True):
        """
        Desc:
            Creates/Saves a Pandas DataFrame object giving the percentage of
            the null data for the original DataFrame columns.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.
        """

        try:
            if not self.__called_from_perform:
                if not self.__check_dataframe(df):

                    if display_print:
                        print("Couldn't create missing values table because"
                              " there is no missing data to display!")
                    return None

            if display_print:
                print("Creating missing values table...")

            if not self.__called_from_perform:
                self.__check_dataframe(df)

            mis_val_table_ren_columns = missing_values_table(df)


            if display_print:
                print(f"Your selected dataframe has {str(df.shape[1])} columns.\n"
                      f"It has {str(mis_val_table_ren_columns.shape[0])} columns missing data.\n")

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

                if not sub_dir:
                    sub_dir = f"{dataset_name}/Tables"

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_table_as_plot(df=df,
                                        df_features=self.__df_features,
                                        filename=filename,
                                        show_index=True,
                                        sub_dir=sub_dir,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        table=mis_val_table_ren_columns)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            plt.close('all')

            if suppress_runtime_errors:
                warnings.warn(
                    f"Missing data table raised an error:\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e



    def __check_dataframe(self,
                          df):
        """
        Args:
            df: pd.Dataframe
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
            df: pd.Dataframe
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