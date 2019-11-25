from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow.utils.pandas_utils import descr_table,value_counts_table
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, SnapshotMismatchError
from eflow._hidden.constants import GRAPH_DEFAULTS
from eflow._hidden.parent_objects import DataAnalysis


import warnings
import random
import numpy as np
from matplotlib import pyplot as plt
import copy
from IPython.display import display
import seaborn as sns
import pandas as pd

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


class FeatureAnalysis(DataAnalysis):

    """
        Analyzes the feature data of a pandas Dataframe object.
        (Ignores null data for displaying data and creates 2d graphics with 2 features.
        In the future I might add 3d graphics with 3 features.)
    """

    def __init__(self,
                 df_features,
                 project_sub_dir="",
                 project_name="Feature Analysis",
                 overwrite_full_path=None,
                 notebook_mode=True):
        """
        Args:

            df_features: DataFrameTypes object from eflow.
                DataFrameTypes object.

            project_sub_dir: string
                Appends to the absolute directory of the output folder

            project_name: string
                Creates a parent or "project" folder in which all sub-directories
                will be inner nested.

            overwrite_full_path: string, None
                Overwrites the path to the parent folder.

            notebook_mode: bool
                If in a python notebook display visualizations in the notebook.
        """

        DataAnalysis.__init__(self,
                              df_features,
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
                         target_feature=None,
                         display_visuals=True,
                         display_print=True,
                         save_file=True,
                         dataframe_snapshot=True,
                         suppress_runtime_errors=True,
                         aggregate_target_feature=True,
                         selected_features=None):
        """
        Desc:
            Performs all public methods that generate visualizations/insights
            about the data.

        Note:
            Pretty much my personal lazy button for running the entire object
            without specifying any method in particular.

        Args:
            df: pd.Dataframe
                Pandas dataframe object

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            target_feature: string or None
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

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            aggregate_target_feature: bool
                Aggregate the data of the target feature if the data is
                non-continuous data.

                Note
                    In the future I will have this also working with continuous
                    data.

            selected_features: collection object of features
                Will only focus on these selected feature's and will ignore
                the other given features.

        Raises:
            If an empty dataframe is passed to this function or if the same
            dataframe is passed to it raise error.
        """
        try:

            self.__called_from_perform = False

            # Raise empty dataframe error
            if df.shape[0] == 0 or np.sum(np.sum(df.isnull()).values) == df.shape[0]:
               raise UnsatisfiedRequirments("Dataframe must contain valid data and "
                                            "not be empty or filled with nulls!")

            # Compare dataframe json file's snapshot to the given dataframe's
            # snapshot
            if dataframe_snapshot:
               df_snapshot = DataFrameSnapshot()
               df_snapshot.check_create_snapshot(df,
                                                 self.__df_features,
                                                 directory_path=self.folder_path,
                                                 sub_dir=f"{dataset_name}/_Extras")

            # Set to true to represent the function call was made with perform
            self.__called_from_perform = True

            if not target_feature:
                target_feature = self.__df_features.get_target_feature()

            # Iterate through features
            for feature_name in df.columns:

               if selected_features and feature_name not in selected_features and feature_name != target_feature:
                   continue

               self.analyze_feature(df,
                                    feature_name,
                                    dataset_name,
                                    target_feature=target_feature,
                                    display_visuals=display_visuals,
                                    save_file=save_file,
                                    dataframe_snapshot=dataframe_snapshot,
                                    suppress_runtime_errors=suppress_runtime_errors,
                                    display_print=display_print)

               # Aggregate data if target feature exists
               if target_feature and feature_name == target_feature and aggregate_target_feature:

                   # -----
                   if target_feature in self.__df_features.get_non_numerical_features() or \
                           target_feature in self.__df_features.get_bool_features():

                       target_feature_values = df[target_feature].value_counts(sort=False).index.to_list()

                       for target_feature_val in target_feature_values:

                           repr_target_feature_val = target_feature_val

                           if target_feature in self.__df_features.get_bool_features():

                               try:
                                   repr_target_feature_val = int(repr_target_feature_val)

                               except ValueError:
                                   continue

                               except TypeError:
                                   continue

                               repr_target_feature_val = str(bool(repr_target_feature_val))

                           # Iterate through all features to generate new graphs for aggregation
                           for f_name in df.columns:

                               if selected_features and f_name not in selected_features and f_name != target_feature:
                                   continue

                               if f_name == target_feature:
                                   continue

                               if display_print:
                                   if repr_target_feature_val:
                                        print(f"Target feature {target_feature} set to {target_feature_val}; also known as {repr_target_feature_val}.")
                                   else:
                                       print(f"Target feature {target_feature} set to {target_feature_val}.")
                               try:
                                   self.analyze_feature(df[df[target_feature] == target_feature_val],
                                                        f_name,
                                                        dataset_name,
                                                        target_feature=target_feature,
                                                        display_visuals=display_visuals,
                                                        save_file=save_file,
                                                        dataframe_snapshot=dataframe_snapshot,
                                                        suppress_runtime_errors=suppress_runtime_errors,
                                                        display_print=display_print,
                                                        sub_dir=f"{dataset_name}/{target_feature}/Where {target_feature} = {repr_target_feature_val}/{f_name}")
                               except Exception as e:
                                   print(f"Error found on feature {f_name}: {e}")

            # If any missed features are picked up...
            missed_features = set(df.columns) ^ self.__df_features.get_all_features()
            if len(missed_features) != 0 and display_print:
                print("Some features were not analyzed by perform analysis!")
                for feature_name in missed_features:
                    print(f"\t\tFeature:{feature_name}")

        # Ensures that called from perform is turned off
        finally:
            self.__called_from_perform = False


    def analyze_feature(self,
                        df,
                        feature_name,
                        dataset_name,
                        target_feature=None,
                        display_visuals=True,
                        display_print=True,
                        sub_dir=None,
                        save_file=True,
                        dataframe_snapshot=True,
                        suppress_runtime_errors=True):
        """
        Desc:
            Generate's all graphic's for that given feature and the relationship
            to the target feature.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object

            feature_name: string
                Specified feature column name.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            target_feature: string
                Will create graphics involving this feature with the main
                feature 'feature_name'.

            display_visuals: string
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            save_file: bool
                Saves file if set to True; doesn't if set to False.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

        Raises:
            Raises error if the json file's snapshot of the given dataframe doesn't
            match the given dataframe.
        """

        colors = self.__get_feature_colors(df,
                                           feature_name)

        # Display colors
        if colors and display_print:
            print(f"Colors:\n{colors}\n")

        if target_feature:

            if target_feature not in self.__df_features.get_all_features():
                raise UnsatisfiedRequirments("Target feature does not exist in pre-defined "
                                             "df_features!")

            target_feature_numerical = target_feature in self.__df_features.get_continuous_numerical_features()
        else:
            target_feature_numerical = False

        two_dim_sub_dir = None
        if sub_dir:
            two_dim_sub_dir = sub_dir
        else:
            if target_feature:
                two_dim_sub_dir = f"{dataset_name}/{target_feature}/Two feature analysis/{target_feature} by {feature_name}"

        # -----
        if feature_name in self.__df_features.get_non_numerical_features() or feature_name in self.__df_features.get_bool_features():

            if len(df[feature_name].value_counts().index) <= 5:
                self.pie_graph(df,
                               feature_name,
                               dataset_name=dataset_name,
                               display_visuals=display_visuals,
                               sub_dir=sub_dir,
                               save_file=save_file,
                               pallete=colors,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               display_print=display_print)

            self.plot_count_graph(df,
                                  feature_name,
                                  dataset_name=dataset_name,
                                  display_visuals=display_visuals,
                                  sub_dir=sub_dir,
                                  save_file=save_file,
                                  dataframe_snapshot=dataframe_snapshot,
                                  suppress_runtime_errors=suppress_runtime_errors,
                                  display_print=display_print)

            if colors:
                self.plot_count_graph(df,
                                      feature_name,
                                      dataset_name=dataset_name,
                                      display_visuals=display_visuals,
                                      sub_dir=sub_dir,
                                      save_file=save_file,
                                      palette=colors,
                                      dataframe_snapshot=dataframe_snapshot,
                                      suppress_runtime_errors=suppress_runtime_errors,
                                      display_print=display_print)

            self.value_counts_table(df,
                                    feature_name,
                                    dataset_name=dataset_name,
                                    display_visuals=display_visuals,
                                    sub_dir=sub_dir,
                                    save_file=save_file,
                                    dataframe_snapshot=dataframe_snapshot,
                                    suppress_runtime_errors=suppress_runtime_errors,
                                    display_print=display_print)

            if target_feature and feature_name != target_feature:

                if target_feature_numerical:

                    if len(set(pd.to_numeric(df[target_feature],
                                         errors='coerce').dropna())) > 1:
                        self.plot_violin_graph(df,
                                               feature_name,
                                               dataset_name=dataset_name,
                                               other_feature_name=target_feature,
                                               display_visuals=display_visuals,
                                               sub_dir=two_dim_sub_dir,
                                               save_file=save_file,
                                               palette=colors,
                                               dataframe_snapshot=dataframe_snapshot,
                                               suppress_runtime_errors=suppress_runtime_errors,
                                               display_print=display_print)
                    self.plot_ridge_graph(df,
                                          feature_name,
                                          dataset_name=dataset_name,
                                          other_feature_name=target_feature,
                                          display_visuals=display_visuals,
                                          sub_dir=two_dim_sub_dir,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          palette=colors,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          display_print=display_print)

        # -----
        elif feature_name in self.__df_features.get_continuous_numerical_features():
            self.plot_distance_graph(df,
                                     feature_name,
                                     dataset_name=dataset_name,
                                     display_visuals=display_visuals,
                                     sub_dir=sub_dir,
                                     save_file=save_file,
                                     dataframe_snapshot=dataframe_snapshot,
                                     suppress_runtime_errors=suppress_runtime_errors,
                                     display_print=display_print)

            self.descr_table(df,
                             feature_name,
                             dataset_name=dataset_name,
                             display_visuals=display_visuals,
                             sub_dir=sub_dir,
                             save_file=save_file,
                             dataframe_snapshot=dataframe_snapshot,
                             suppress_runtime_errors=suppress_runtime_errors,
                             display_print=display_print)

            if target_feature and feature_name != target_feature:

                if target_feature_numerical:
                    # Jointplot
                    pass

                else:
                    if len(set(pd.to_numeric(df[feature_name],
                                             errors='coerce').dropna())) > 1:
                        self.plot_violin_graph(df,
                                               target_feature,
                                               dataset_name=dataset_name,
                                               other_feature_name=feature_name,
                                               display_visuals=display_visuals,
                                               sub_dir=two_dim_sub_dir,
                                               save_file=save_file,
                                               palette=colors,
                                               dataframe_snapshot=dataframe_snapshot,
                                               suppress_runtime_errors=suppress_runtime_errors,
                                               display_print=display_print)

                    self.plot_ridge_graph(df,
                                          target_feature,
                                          dataset_name=dataset_name,
                                          other_feature_name=feature_name,
                                          display_visuals=display_visuals,
                                          sub_dir=two_dim_sub_dir,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          palette=colors,
                                          display_print=display_print)
        if display_print:
            print("\n\n")

    def plot_distance_graph(self,
                            df,
                            feature_name,
                            dataset_name,
                            display_visuals=True,
                            display_print=True,
                            filename=None,
                            sub_dir=None,
                            save_file=True,
                            dataframe_snapshot=True,
                            suppress_runtime_errors=True,
                            figsize=GRAPH_DEFAULTS.FIGSIZE,
                            bins=None,
                            norm_hist=True,
                            hist=True,
                            kde=True,
                            colors=None,
                            fit=None,
                            fit_kws=None):
        """
        Desc:
            Display a distance plot and save the graph in the correct directory.

        Args:
            df: pd.Dataframe
                Pandas dataframe object

            feature_name: string
                Specified feature column name.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                Name to give the file.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

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

            figsize: tuple
                The given size of the plot.

            bins: int
                Specification of hist bins, or None to use Freedman-Diaconis rule.

            norm_hist: bool
                If True, the histogram height shows a density rather than a count. This is implied if a KDE or fitted density is plotted.

            hist: bool
                Whether to plot a (normed) histogram.

            kde: bool
                Whether to plot a gaussian kernel density estimate.

            colors : matplotlib color
                Color to plot everything but the fitted curve in.

            fit: functional method
                An object with fit method, returning a tuple that can be passed
                to a pdf method a positional arguments following an grid of
                values to evaluate the pdf on.

            fit_kws : dictionaries, optional
                Keyword arguments for underlying plotting functions.

            Credit to seaborn's author:
            Michael Waskom
            Git username: mwaskom
            Doc Link: http://tinyurl.com/ycco2hok

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Distance plot graph couldn't be generated because " +
                    f"there is only missing data to display in {feature_name}!")

            if display_print:
                print(f"Generating graph for distance plot on {feature_name}")

            feature_values = pd.to_numeric(df[feature_name].dropna(),
                                           errors='coerce').dropna()

            if not len(feature_values):
                raise ValueError(
                    f"The given feature {feature_name} doesn't seem to convert to a numeric vector.")

            # Closes up any past graph info
            plt.close('all')

            # Set foundation graph info
            sns.set(style="whitegrid")
            plt.figure(figsize=figsize)
            plt.title("Distance Plot: " + feature_name)

            # Create seaborn graph
            sns.distplot(feature_values,
                         bins=bins,
                         hist=hist,
                         kde=kde,
                         fit=fit,
                         fit_kws=fit_kws,
                         color=colors,
                         norm_hist=norm_hist)

            # Pass a default name if needed
            if not filename:
                filename = f"Distance plot graph on {feature_name}"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors)


            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close('all')

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close('all')
            if suppress_runtime_errors:
                warnings.warn(
                    f"Distance plot graph throw an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e


    def plot_violin_graph(self,
                          df,
                          feature_name,
                          dataset_name,
                          other_feature_name=None,
                          display_visuals=True,
                          display_print=True,
                          filename=None,
                          sub_dir=None,
                          save_file=True,
                          dataframe_snapshot=True,
                          suppress_runtime_errors=True,
                          figsize=GRAPH_DEFAULTS.FIGSIZE,
                          order=None,
                          cut=2,
                          scale='area',
                          gridsize=100,
                          width=0.8,
                          palette=None,
                          saturation=0.75):
        """
        Desc:
            Display a violin plot and save the graph in the correct directory.

        Args:
            df: pd.Dataframe
                Pandas dataframe object

            feature_name: string
                Specified feature column name to compare to y.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            other_feature_name: string
                Specified feature column name to compare to x.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            filename: string
                Name to give the file.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

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

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            figsize: tuple
                Size of the given plot.

            order: lists of strings
                Order to plot the categorical levels in, otherwise the levels
                are inferred from the data objects.

            cut: float
                Distance, in units of bandwidth size, to extend the density
                past the extreme datapoints. Set to 0 to limit the violin range
                within the range of the observed data.
                (i.e., to have the same effect as trim=True in ggplot.)

            scale: string
                {area, count, width}
                The method used to scale the width of each violin. If area,
                each violin will have the same area. If count, the width of the
                violins will be scaled by the number of observations in that
                bin. If width, each violin will have the same width.

            gridsize: int
                Number of points in the discrete grid used to compute the kernel density estimate.

            width: float
                Width of a full element when not using hue nesting, or width of
                all the elements for one level of the major grouping variable.

            palette: dict or string
                Colors to use for the different levels of the hue variable.
                Should be something that can be interpreted by color_palette(),
                or a dictionary mapping hue levels to matplotlib colors.

            saturation: float
                Proportion of the original saturation to draw colors at. Large
                patches often look better with slightly desaturated colors, but
                set this to 1 if you want the plot colors to perfectly match
                the input color spec.

            Credit to seaborn's author:
            Michael Waskom
            Git username: mwaskom
            Doc link: http://tinyurl.com/y3hxxzgv

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:
            # Error check and create title/part of default file name
            found_features = []
            feature_title = ""
            for feature in (feature_name, other_feature_name):
                if feature:
                    if np.sum(df[feature].isnull()) == df.shape[0]:
                        raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                              f"there is only missing data to display in {feature}!")

                    found_features.append(feature)
                    if len(found_features) == 1:
                        feature_title = f"{feature}"
                    else:
                        feature_title += f" by {feature}"


            if not len(found_features):
                raise UnsatisfiedRequirments("Both x and y feature's are type 'None'. Please pass at least one feature.")

            del found_features

            if display_print:
                print("Generating graph violin graph on " + feature_title)

            # Closes up any past graph info
            plt.close('all')

            # Set plot structure
            plt.figure(figsize=figsize)
            plt.title("Violin Plot: " + feature_title)

            feature_values = pd.to_numeric(df[other_feature_name],
                                           errors='coerce').dropna()

            if not len(feature_values):
                raise ValueError("The y feature must contain numerical features.")

            x_values = copy.deepcopy(df[feature_name])

            # if feature_name in self.__df_features.get_bool_features():
            #     x_values = pd.to_numeric(x_values,
            #                              errors='ignore')
            #
            #     x_values = ['True' if val == 1 else 'False'
            #                 if val == 0 else val
            #                 for val in x_values]


            sns.violinplot(x=x_values,
                           y=feature_values,
                           order=order,
                           cut=cut,
                           scale=scale,
                           gridsize=gridsize,
                           width=width,
                           palette=palette,
                           saturation=saturation)

            # Pass a default name if needed
            if not filename:
                filename = f"Violin plot graph on {feature_title}."

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors)

            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close('all')

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close('all')
            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot violin graph an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def plot_count_graph(self,
                         df,
                         feature_name,
                         dataset_name,
                         display_visuals=True,
                         display_print=True,
                         filename=None,
                         sub_dir=None,
                         save_file=True,
                         dataframe_snapshot=True,
                         suppress_runtime_errors=True,
                         figsize=GRAPH_DEFAULTS.FIGSIZE,
                         flip_axis=False,
                         palette="PuBu"):
        """
        Desc:
            Display a barplot with color ranking from a feature's value counts
            from the seaborn libary and save the graph in the correct directory
            structure.

        Args:
            df: pd.Dataframe
                Pandas dataframe object.

            feature_name: string
                Specified feature column name.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                Name to give the file.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

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

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            figsize: tuple
                Size for the given plot.

            flip_axis: bool
                Flip the axis the ploting axis from x to y if set to 'True'.

            palette: dict or string
                String representation of color pallete for ranking from seaborn's pallete.

            Credit to seaborn's author:
            Michael Waskom
            Git username: mwaskom
            Link: http://tinyurl.com/y4pzrgcf

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Count plot graph couldn't be generated because " +
                    f"there is only missing data to display in {feature_name}!")

            if display_print:
                print(f"Count plot graph on {feature_name}")

            # Closes up any past graph info
            plt.close('all')

            # Set graph info
            plt.figure(figsize=figsize)
            sns.set(style="whitegrid")
            plt.title("Category Count Plot: " + feature_name)

            value_counts = df[feature_name].dropna().value_counts(sort=True)

            feature_values,counts = value_counts.index, value_counts.values
            del value_counts

            # Find and rank values based on counts for color variation of the graph
            if not palette:
                palette = "PuBu"

            if isinstance(palette,str):
                rank_list = np.argsort(-np.array(counts)).argsort()
                pal = sns.color_palette(palette, len(counts))
                palette = np.array(pal[::-1])[rank_list]

            plt.clf()

            if feature_name in self.__df_features.get_bool_features():

                i = 0
                for val in feature_values:
                    try:
                        feature_values[i] = float(val)
                    except:
                        pass

                feature_values = [bool(val) if val == 0 or val == 1 else val
                                  for val in feature_values]

            # Flip the graph for visual flare
            if flip_axis:
                ax = sns.barplot(x=counts,
                                 y=feature_values,
                                 palette=palette,
                                 order=feature_values)
            else:
                ax = sns.barplot(x=feature_values,
                                 y=counts,
                                 palette=palette,
                                 order=feature_values)

            # Labels for numerical count of each bar
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2.,
                        height + 3,
                        '{:1}'.format(height),
                        ha="center")

            # Pass a default name if needed
            if not filename:
                filename = f"Count plot graph on {feature_name}"

                if isinstance(palette,np.ndarray):
                    filename += " with count color ranking."

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors)

            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close('all')

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close('all')
            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot count graph raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def pie_graph(self,
                  df,
                  feature_name,
                  dataset_name,
                  display_visuals=True,
                  display_print=True,
                  filename=None,
                  sub_dir=None,
                  save_file=True,
                  dataframe_snapshot=True,
                  suppress_runtime_errors=True,
                  figsize=GRAPH_DEFAULTS.FIGSIZE,
                  pallete=None):
        """
        Desc:
            Display a pie graph and save the graph in the correct directory.

        Args:
           df:
               Pandas DataFrame object.

           feature_name:
               Specified feature column name.

           dataset_name:
               The dataset's name; this will create a sub-directory in which your
               generated graph will be inner-nested in.

           display_visuals:
               Boolean value to whether or not to display visualizations.

           display_print: bool
                Determines whether or not to print function's embedded print
                statements.

           filename:
               If set to 'None' will default to a pre-defined string;
               unless it is set to an actual filename.

           sub_dir:
               Specify the sub directory to append to the pre-defined folder path.

           save_file:
               Boolean value to whether or not to save the file.

           dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

           suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

           figsize: tuple
                Size of the plot.

           pallete: dict or string
                Dictionary of all feature values to hex color values.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Pie graph couldn't be generated because " +
                      f"there is only missing data to display in {feature_name}!")

            if display_print:
                print(f"Pie graph on {feature_name}")

            # Closes up any past graph info
            plt.close('all')

            # Find value counts
            value_counts = df[feature_name].dropna().value_counts(sort=False)
            feature_values = value_counts.index.tolist()
            value_count_list = value_counts.values.tolist()

            # Explode the part of the pie graph that is the maximum of the graph
            explode_array = [0] * len(feature_values)
            explode_array[np.array(value_count_list).argmax()] = .03

            color_list = None

            if isinstance(pallete,dict):
                color_list = []
                for value in tuple(feature_values):
                    try:
                        color_list.append(pallete[value])
                    except KeyError:
                        raise KeyError(f"The given value '{value}' in feature '{feature_name}'"
                                       + " was not found in the passed color dict.")

            plt.figure(figsize=figsize)

            if feature_name in self.__df_features.get_bool_features():

                i = 0
                for val in feature_values:
                    try:
                        feature_values[i] = float(val)
                    except:
                        pass
                feature_values = [bool(val) if val == 0 or val == 1 else val
                                  for val in feature_values]

            # Plot pie graph
            plt.pie(
                tuple(value_count_list),
                labels=tuple(feature_values),
                shadow=False,
                colors=color_list,
                explode=tuple(explode_array),
                startangle=90,
                autopct='%1.1f%%',
            )

            # Set foundation graph info
            fig = plt.gcf()
            plt.title("Pie Chart: " + feature_name)
            plt.legend(fancybox=True)
            plt.axis('equal')
            plt.tight_layout()

            # Pass a default name if needed
            if not filename:
                filename = f"Pie graph on {feature_name}"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors)


            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close('all')

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            if suppress_runtime_errors:
                warnings.warn(
                    f"Pie graph raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def plot_ridge_graph(self,
                         df,
                         feature_name,
                         dataset_name,
                         other_feature_name,
                         display_visuals=True,
                         display_print=True,
                         filename=None,
                         sub_dir=None,
                         save_file=True,
                         dataframe_snapshot=True,
                         suppress_runtime_errors=True,
                         figsize=GRAPH_DEFAULTS.FIGSIZE,
                         palette=None):
        """
        Desc:
            Display a ridge plot and save the graph in the correct directory.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object.

            feature_name: string
                Specified feature column name.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            other_feature_name: string
                Feature to compare to.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            filename: string
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

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

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            figsize: tuple
                Tuple object to represent the plot/image's size.

            palette: dict or string
                Dictionary of all feature values to hex color values.

        Note:
            A large part of this was taken from: http://tinyurl.com/tuou2cn

        Raises:
           Raises error if the feature data is filled with only nulls or if
           the json file's snapshot of the given dataframe doesn't match the
           given dataframe.
        """
        try:
            if display_print:
                print(f"Ridge plot graph on {feature_name} by {other_feature_name}.")

            sns.set(style="white",
                    rc={"axes.facecolor": (0, 0, 0, 0)})

            # Temporarily turn off chained assignments
            chained_assignment = pd.options.mode.chained_assignment
            pd.options.mode.chained_assignment = None

            tmp_df = copy.deepcopy(df[[feature_name,other_feature_name]])
            tmp_df[other_feature_name] = pd.to_numeric(tmp_df[other_feature_name],
                                                       errors='coerce')

            # if feature_name in self.__df_features.get_bool_features():
            #
            #     tmp_df[feature_name] = pd.to_numeric(tmp_df[feature_name],
            #                                          errors='ignore')
            #     tmp_df[feature_name] = ['True' if val == 1 else 'False'
            #                               if val == 0 else val
            #                             for val in tmp_df[feature_name]]

            tmp_df.dropna(inplace=True)

            # Remove any values that only return a single value back
            for val in set(tmp_df[feature_name]):
                if len(tmp_df[other_feature_name][tmp_df[feature_name] == val]) <= 1:
                    tmp_df = tmp_df[tmp_df[feature_name] != val]

            pd.options.mode.chained_assignment = chained_assignment
            del chained_assignment

            sns.set(style="white",
                    rc={"axes.facecolor": (0, 0, 0, 0)})

            if not palette:
                palette = sns.cubehelix_palette(10, rot=-.20, light=.7)

            # Initialize the FacetGrid object
            g = sns.FacetGrid(tmp_df,
                              row=feature_name,
                              hue=feature_name,
                              aspect=15,
                              height=.4,
                              palette=palette)

            # Draw the densities in a few steps
            g.map(sns.kdeplot,
                  other_feature_name,
                  clip_on=False,
                  shade=True,
                  alpha=1,
                  lw=1.5,
                  bw=.2)

            g.map(sns.kdeplot,
                  other_feature_name,
                  clip_on=False,
                  color="w",
                  lw=2,
                  bw=.2)

            g.map(plt.axhline,
                  y=0,
                  lw=2,
                  clip_on=False)

            # Define and use a simple function to label the plot in axes coordinates
            def label(x, color, label):
                ax = plt.gca()
                ax.text(-.1,
                        .2,
                        label,
                        fontweight="bold",
                        color=color,
                        ha="left",
                        va="center",
                        transform=ax.transAxes)

            g.map(label, other_feature_name)

            # Set the subplots to overlap
            g.fig.subplots_adjust(hspace=-.25)

            # Remove axes details that don't play well with overlap
            g.set_titles("")
            g.set(yticks=[])
            g.despine(bottom=True, left=True)
            g.fig.set_size_inches(10, 10, forward=True)
            g.fig.suptitle(f'{feature_name} by {other_feature_name}')
            plt.figure(figsize=figsize)

            # Pass a default name if needed
            if not filename:
                filename = f"Ridge plot graph on {feature_name} by {other_feature_name}"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:


                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_plot(df=df,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors)

            if self.__notebook_mode and display_visuals:
                plt.show()

            plt.close('all')

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close('all')
            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot ridge graph raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e


    def plot_jointplot_graph(self,
                             df,
                             feature_name,
                             dataset_name,
                             other_feature_name):
        raise ValueError("NOT READY FOR USE YET!!!!!!")


    def value_counts_table(self,
                           df,
                           feature_name,
                           dataset_name,
                           display_visuals=True,
                           display_print=True,
                           filename=None,
                           sub_dir=None,
                           save_file=True,
                           dataframe_snapshot=True,
                           suppress_runtime_errors=True):
        """
        Args:
            df: pd.Dataframe
                Pandas DataFrame object

            feature_name: string
                Specified feature column name.

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

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            save_file: bool
                Saves file if set to True; doesn't if set to False.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

        Desc:
            Creates/Saves a pandas dataframe of value counts of a dataframe.

            Note -
                Creates a png of the table.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:
            # Check if feature has only null data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Values count table couldn't be generated because " +
                                             f"there is only missing data to display in {feature_name}!")

            if display_print:
                print(f"Creating value counts table for feature {feature_name}.")

            # -----
            val_counts_df = value_counts_table(df,
                                               feature_name)

            if self.__notebook_mode:
                if display_visuals:
                    display(val_counts_df)
            else:
                if display_visuals:
                    print(val_counts_df)

            # Pass a default name if needed
            if not filename:
                filename = f"{feature_name} Value Counts Table"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_table_as_plot(df=df,
                                        filename=filename,
                                        sub_dir=sub_dir,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        table=val_counts_df)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close('all')
            if suppress_runtime_errors:
                warnings.warn(
                    f"Value count table raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def descr_table(self,
                    df,
                    feature_name,
                    dataset_name,
                    display_visuals=True,
                    display_print=True,
                    filename=None,
                    sub_dir=None,
                    save_file=True,
                    dataframe_snapshot=True,
                    suppress_runtime_errors=True):
        """
        Desc:
            Creates/Saves a pandas dataframe of features and their found types
            in the dataframe.

            Note
                Creates a png of the table.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object

            feature_name: string
                Specified feature column name.

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

            sub_dir: string
                Specify the sub directory to append to the pre-defined folder path.

            save_file: bool
                Saves file if set to True; doesn't if set to False.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:
            # Check if dataframe has only null data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                      f"there is only missing data to display in {feature_name}!")

            if display_print:
                print(f"Creating data description table for {feature_name}")

            desc_df = descr_table(df,
                                  feature_name,
                                  to_numeric=True)

            if self.__notebook_mode:
                if display_visuals:
                    display(desc_df)
            else:
                if display_visuals:
                    print(desc_df)

            # Pass a default name if needed
            if not filename:
                filename = f"{feature_name} Description Table"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_table_as_plot(df=df,
                                        filename=filename,
                                        sub_dir=sub_dir,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        table=desc_df)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            plt.close('all')
            if suppress_runtime_errors:
                warnings.warn(
                    f"Descr table raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

    def __get_feature_colors(self,
                             df,
                             feature_name):
        """
        Desc:
            Creates a dict object of all possible feature values with their
            associated colors.

            Note
                Any unknown feature values that aren't declared
                by df_features are given a default color from the constants
                section of the project. Goes up to 20 different colors unitl
                colors is init to None.

        Args:
            df: pd.Dataframe
                Pandas DataFrame object

            feature_name: string
                Specified feature column name.

        Returns:
            Gives back a dictionary object of all possible feature values
            with their associated colors.
        """


        colors = self.__df_features.get_feature_colors(feature_name)
        feature_value_representation = self.__df_features.get_feature_value_representation()

        if colors:
            if isinstance(colors, dict):
                feature_values = df[feature_name].value_counts(
                    sort=False).keys().to_list()
                decoder = self.__df_features.get_label_decoder()

                # Add color feature value for decoders values
                if feature_name in decoder.keys():
                    for cat, val in decoder[feature_name].items():

                        if cat in colors.keys():
                            hex_code = colors[cat]
                            colors[decoder[feature_name][cat]] = hex_code

                        elif val in colors.keys():
                            hex_code = colors[val]
                            colors[cat] = hex_code

                # Add color feature value for different value representation
                if feature_name in feature_value_representation.keys():
                    for val in feature_value_representation[
                        feature_name].keys():
                        if val in colors.keys():
                            hex_code = colors[val]
                            colors[feature_value_representation[
                                feature_name][val]] = hex_code

                i = 0
                for value in feature_values:
                    if value not in colors.keys():
                        colors[value] = \
                        GRAPH_DEFAULTS.DEFINED_LIST_OF_RANDOM_COLORS[i]
                        i += 1

                        if i == len(
                                GRAPH_DEFAULTS.DEFINED_LIST_OF_RANDOM_COLORS):
                            colors = None
        return colors