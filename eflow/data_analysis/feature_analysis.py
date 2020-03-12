from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow.utils.pandas_utils import descr_table,value_counts_table
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments, SnapshotMismatchError
from eflow._hidden.constants import GRAPH_DEFAULTS
from eflow._hidden.parent_objects import DataAnalysis
from eflow.utils.pandas_utils import check_if_feature_exists, generate_meta_data, generate_entropy_table, feature_correlation_table, average_feature_correlation_table
from eflow.utils.sys_utils import dict_to_json_file, pickle_object_to_file, create_dir_structure

import warnings
import random
import numpy as np
from matplotlib import pyplot as plt
import copy
from IPython.display import display
from eflow.utils.pandas_utils import auto_binning
import seaborn as sns
import pandas as pd
from scipy import stats

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
                 notebook_mode=False):
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
                         target_features=None,
                         display_visuals=True,
                         display_print=True,
                         save_file=True,
                         dataframe_snapshot=True,
                         suppress_runtime_errors=True,
                         figsize=GRAPH_DEFAULTS.FIGSIZE,
                         aggregate_target_feature=True,
                         selected_features=None,
                         extra_tables=True,
                         statistical_analysis_on_aggregates=True):
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

            target_features: collection of strings or None
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

            extra_tables: bool
                When handling two types of features if set to true this will
                    generate any extra tables that might be helpful.
                    Note -
                        These graphics may create duplicates if you already applied
                        an aggregation in 'perform_analysis'

            aggregate_target_feature: bool
                Aggregate the data of the target feature if the data is
                non-continuous data.

                Note
                    In the future I will have this also working with continuous
                    data.

            selected_features: collection object of features
                Will only focus on these selected feature's and will ignore
                the other given features.

            statistical_analysis_on_aggregates: bool
                If set to true then the function 'statistical_analysis_on_aggregates'
                will run; which aggregates the data of the target feature either
                by discrete values or by binning/labeling continuous data.

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

            generate_meta_data(df,
                               self.folder_path,
                               f"{dataset_name}" + "/_Extras")

            generate_entropy_table(df,
                                   self.__df_features,
                                   self.folder_path,
                                   f"{dataset_name}" + "/_Extras/Statistics")

            corr_df = feature_correlation_table(df)
            self.save_table_as_plot(df=df,
                                    table=corr_df,
                                    show_index=True,
                                    format_float_pos=7,
                                    df_features=self.__df_features,
                                    filename="Correlation Table",
                                    sub_dir=f"{dataset_name}" + "/_Extras/Statistics",
                                    dataframe_snapshot=False,
                                    suppress_runtime_errors=suppress_runtime_errors,
                                    meta_data=False)


            corr_df = average_feature_correlation_table(df)
            self.save_table_as_plot(df=df,
                                    table=corr_df,
                                    show_index=True,
                                    format_float_pos=7,
                                    df_features=self.__df_features,
                                    filename="Average Correlation Table",
                                    sub_dir=f"{dataset_name}" + "/_Extras/Statistics",
                                    dataframe_snapshot=False,
                                    suppress_runtime_errors=suppress_runtime_errors,
                                    meta_data=False)

            # Init color ranking fo plot
            # Ref: http://tinyurl.com/ydgjtmty
            plt.figure(figsize=(13, 10))
            pal = sns.color_palette("GnBu_d", len(corr_df["Average Correlations"]))
            rank = np.array(corr_df["Average Correlations"]).argsort().argsort()
            ax = sns.barplot(y=corr_df.index.tolist(), x=corr_df["Average Correlations"],
                             palette=np.array(pal[::-1])[rank])
            plt.xticks(rotation=0, fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel("Features", fontsize=20, labelpad=20)
            plt.ylabel("Correlation Average", fontsize=20, labelpad=20)
            plt.title("Average Feature Correlation", fontsize=15)
            self.save_plot(df=df,
                           df_features=self.__df_features,
                           filename="Average Correlation Rank Graph",
                           sub_dir=f"{dataset_name}" + "/_Extras/Statistics",
                           dataframe_snapshot=False,
                           suppress_runtime_errors=suppress_runtime_errors,
                           meta_data=False)

            plt.close("all")

            del corr_df

            # Set to true to represent the function call was made with perform
            self.__called_from_perform = True

            if isinstance(target_features,str):
                target_features = {target_features}

            if not target_features:
                target_features = {None}

            if isinstance(target_features,list):
                target_features = set(target_features)

            # Iterate through all target features
            for target_feature in target_features:

                # Iterate through all dataframe features
                for feature_name in df.columns:

                   # Only compare selected features if user specfied features
                   if selected_features and feature_name not in selected_features and feature_name != target_feature:
                       continue

                   # Ignore if the feature is found to be purely null
                   if feature_name in self.__df_features.null_only_features():
                       continue

                   self.analyze_feature(df,
                                        feature_name,
                                        dataset_name,
                                        target_feature=target_feature,
                                        display_visuals=display_visuals,
                                        save_file=save_file,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        figsize=figsize,
                                        display_print=display_print,
                                        extra_tables=extra_tables)

                   # Aggregate data if target feature exists
                   if target_feature and feature_name == target_feature and aggregate_target_feature:

                       # -----
                       if target_feature in self.__df_features.non_numerical_features() or \
                               target_feature in self.__df_features.bool_features():

                           target_feature_values = df[target_feature].value_counts(sort=False).index.to_list()

                           # Begin aggregation
                           for target_feature_val in target_feature_values:

                               repr_target_feature_val = target_feature_val

                               # Convert to best bool representation of value
                               if target_feature in self.__df_features.bool_features():

                                   try:
                                       repr_target_feature_val = bool(int(repr_target_feature_val))

                                   except ValueError:
                                       continue

                                   except TypeError:
                                       continue

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
                                                            figsize=figsize,
                                                            display_print=display_print,
                                                            sub_dir=f"{dataset_name}/{target_feature}/Where {target_feature} = {repr_target_feature_val}/{f_name}",
                                                            extra_tables=False)
                                   except Exception as e:
                                       print(f"Error found on feature {f_name}: {e}")

                # If any missed features are picked up...
                missed_features = set(df.columns) ^ self.__df_features.all_features()
                if len(missed_features) != 0 and display_print:
                    print("Some features were not analyzed by perform analysis!")
                    for feature_name in missed_features:
                        print(f"\t\tFeature:{feature_name}")

            if statistical_analysis_on_aggregates and target_feature:
                self.statistical_analysis_on_aggregates(df,
                                                        target_features,
                                                        dataset_name,
                                                        dataframe_snapshot=False)
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
                        suppress_runtime_errors=True,
                        figsize=GRAPH_DEFAULTS.FIGSIZE,
                        extra_tables=True):
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

            extra_tables: bool
                When handling two types of features if set to true this will
                generate any extra tables that might be helpful.
                Note -
                    These graphics may create duplicates if you already applied
                    an aggregation in 'perform_analysis'

        Raises:
            Raises error if the json file's snapshot of the given dataframe doesn't
            match the given dataframe.
        """

        # -----
        check_if_feature_exists(df,
                                feature_name)

        colors = self.__get_feature_colors(df,
                                           feature_name)

        # Display colors
        if colors and display_print:
            print(f"Colors:\n{colors}\n")


        # Check if feature exist in df_features and by extension the dataframe
        if target_feature:

            if target_feature not in self.__df_features.all_features():
                raise UnsatisfiedRequirments("Target feature does not exist in pre-defined "
                                             "df_features!")
            if target_feature not in df.columns:
                raise UnsatisfiedRequirments("Target feature does not exist in "
                                             "the dataframe!")

        if feature_name not in self.__df_features.all_features():
            raise UnsatisfiedRequirments(
                "Feature name does not exist in pre-defined "
                "df_features!")
        if feature_name not in df.columns:
            raise UnsatisfiedRequirments("Feature name does not exist in "
                                         "the dataframe!")
        # Generate sub directory structure for plots involving two features
        two_dim_sub_dir = None
        if sub_dir:
            two_dim_sub_dir = sub_dir
        else:
            if target_feature:
                two_dim_sub_dir = f"{dataset_name}/{target_feature}/Two feature analysis/{target_feature} by {feature_name}"

        # -----
        if feature_name in self.__df_features.non_numerical_features() or feature_name in self.__df_features.bool_features():

            # Pie graph's should only have less than or equal to six.
            # (The function can handle ample more than this; just stylistically)
            if len(df[feature_name].value_counts().index) <= 5:
                self.plot_pie_graph(df,
                                    feature_name,
                                    dataset_name=dataset_name,
                                    display_visuals=display_visuals,
                                    sub_dir=sub_dir,
                                    save_file=save_file,
                                    pallete=colors,
                                    dataframe_snapshot=dataframe_snapshot,
                                    suppress_runtime_errors=suppress_runtime_errors,
                                    figsize=figsize,
                                    display_print=display_print)

            # Count plot without colors
            self.plot_count_graph(df,
                                  feature_name,
                                  dataset_name=dataset_name,
                                  display_visuals=display_visuals,
                                  sub_dir=sub_dir,
                                  save_file=save_file,
                                  dataframe_snapshot=dataframe_snapshot,
                                  suppress_runtime_errors=suppress_runtime_errors,
                                  display_print=display_print)

            # Count plot with colors
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
                                      figsize=figsize,
                                      display_print=display_print)

            # Generate value counts table
            self.value_counts_table(df,
                                    feature_name,
                                    dataset_name=dataset_name,
                                    display_visuals=display_visuals,
                                    sub_dir=sub_dir,
                                    save_file=save_file,
                                    dataframe_snapshot=dataframe_snapshot,
                                    suppress_runtime_errors=suppress_runtime_errors,
                                    display_print=display_print)

        # -----
        elif feature_name in self.__df_features.continuous_numerical_features():

            # Plot distance plot graph
            self.plot_distance_graph(df,
                                     feature_name,
                                     dataset_name=dataset_name,
                                     display_visuals=display_visuals,
                                     sub_dir=sub_dir,
                                     save_file=save_file,
                                     dataframe_snapshot=dataframe_snapshot,
                                     suppress_runtime_errors=suppress_runtime_errors,
                                     figsize=figsize,
                                     display_print=display_print)

            # Create description table
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

            # Simplified conditional check for finding type relationship between the two features
            num_features = []
            non_num_features = []

            if target_feature in self.__df_features.continuous_numerical_features():
                num_features.append(target_feature)
            elif target_feature in self.__df_features.datetime_features():
                pass
            elif target_feature not in self.__df_features.continuous_numerical_features():
                non_num_features.append(target_feature)

            if feature_name in self.__df_features.continuous_numerical_features():
                num_features.append(feature_name)
            elif feature_name in self.__df_features.datetime_features():
                pass
            elif feature_name not in self.__df_features.continuous_numerical_features():
                non_num_features.append(feature_name)

            # Two different types of features (numerical and non-numerical)
            if len(num_features) == 1 and len(non_num_features) == 1:

                # Extract out feature name's to better named variables for sanity
                numerical_feature = num_features.pop()
                non_numerical_feature = non_num_features.pop()

                # Generate violin
                self.plot_violin_graph(df,
                                       non_numerical_feature,
                                       dataset_name=dataset_name,
                                       other_feature_name=numerical_feature,
                                       display_visuals=display_visuals,
                                       sub_dir=two_dim_sub_dir,
                                       save_file=save_file,
                                       palette=colors,
                                       dataframe_snapshot=dataframe_snapshot,
                                       suppress_runtime_errors=suppress_runtime_errors,
                                       figsize=figsize,
                                       display_print=display_print)

                # Generate ridge graph
                self.plot_ridge_graph(df,
                                      non_numerical_feature,
                                      dataset_name=dataset_name,
                                      other_feature_name=numerical_feature,
                                      display_visuals=display_visuals,
                                      sub_dir=two_dim_sub_dir,
                                      save_file=save_file,
                                      dataframe_snapshot=dataframe_snapshot,
                                      palette=colors,
                                      suppress_runtime_errors=suppress_runtime_errors,
                                      figsize=figsize,
                                      display_print=display_print)

                # Generate tables based on the aggregation of the non-numerical feature
                if extra_tables:
                    for val in df[non_numerical_feature].unique():

                        if display_print:
                            print(f"Where {non_numerical_feature} = {val}")

                        # Create new sub dir based on the aggregation
                        two_dim_desc_sub_dir = copy.deepcopy(
                            two_dim_sub_dir)
                        if not two_dim_desc_sub_dir:
                            two_dim_desc_sub_dir = ""
                        two_dim_desc_sub_dir += "/" + str(val)

                        # Create new dataframe on aggregated value and check for nans
                        tmp_df = df[df[non_numerical_feature] == val]
                        if np.sum(tmp_df[numerical_feature].isnull()) != \
                                tmp_df.shape[0]:

                            self.descr_table(df=tmp_df,
                                             feature_name=numerical_feature,
                                             dataset_name=dataset_name,
                                             display_visuals=display_visuals,
                                             display_print=display_print,
                                             sub_dir=two_dim_desc_sub_dir,
                                             dataframe_snapshot=False)
                            if display_print:
                                print("\n")

                        del tmp_df

            elif len(non_num_features) == 2:

                # Generate tables based on the aggregation of the non-numerical feature
                if extra_tables:
                    for val in df[feature_name].dropna().unique():

                        if display_print:
                            print(f"Where {feature_name} = {val}")

                        # Create new sub dir based on the aggregation
                        two_dim_desc_sub_dir = copy.deepcopy(
                            two_dim_sub_dir)
                        if not two_dim_desc_sub_dir:
                            two_dim_desc_sub_dir = ""
                        two_dim_desc_sub_dir += "/" + str(val)

                        # Create new dataframe on aggregated value and check for nans
                        tmp_df = df[df[feature_name] == val]
                        if np.sum(tmp_df[target_feature].isnull()) != \
                                tmp_df.shape[0]:
                            self.value_counts_table(df=df[df[feature_name] == val],
                                                    feature_name=target_feature,
                                                    dataset_name=dataset_name,
                                                    display_visuals=display_visuals,
                                                    display_print=display_print,
                                                    sub_dir=two_dim_desc_sub_dir,
                                                    dataframe_snapshot=False)
                            if display_print:
                                print("\n")

                self.group_by_feature_value_count_table(df,
                                                        feature_name,
                                                        dataset_name=dataset_name,
                                                        other_feature_name=target_feature,
                                                        display_visuals=display_visuals,
                                                        sub_dir=two_dim_sub_dir,
                                                        save_file=save_file,
                                                        dataframe_snapshot=dataframe_snapshot,
                                                        suppress_runtime_errors=suppress_runtime_errors,
                                                        display_print=display_print)
                self.plot_multi_bar_graph(df,
                                          feature_name,
                                          dataset_name=dataset_name,
                                          other_feature_name=target_feature,
                                          display_visuals=display_visuals,
                                          sub_dir=two_dim_sub_dir,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          figsize=figsize,
                                          display_print=display_print,
                                          stacked=False)

                self.plot_multi_bar_graph(df,
                                          feature_name,
                                          dataset_name=dataset_name,
                                          other_feature_name=target_feature,
                                          display_visuals=display_visuals,
                                          sub_dir=two_dim_sub_dir,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          figsize=figsize,
                                          display_print=display_print,
                                          stacked=True)


            elif len(num_features) == 2:

                # Generate jointplot graph with scatter and kde
                self.plot_jointplot_graph(df,
                                          feature_name,
                                          dataset_name=dataset_name,
                                          other_feature_name=target_feature,
                                          display_visuals=display_visuals,
                                          sub_dir=two_dim_sub_dir,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          color=colors,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          figsize=figsize,
                                          display_print=display_print)

                # Generate jointplot graph with kde
                self.plot_jointplot_graph(df,
                                          feature_name,
                                          dataset_name=dataset_name,
                                          other_feature_name=target_feature,
                                          display_visuals=display_visuals,
                                          sub_dir=two_dim_sub_dir,
                                          save_file=save_file,
                                          dataframe_snapshot=dataframe_snapshot,
                                          color=colors,
                                          suppress_runtime_errors=suppress_runtime_errors,
                                          figsize=figsize,
                                          display_print=display_print,
                                          kind="kde")
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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            # Error check
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
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

            if self.__notebook_mode and display_visuals:
                plt.show()

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            if suppress_runtime_errors:
                warnings.warn(
                    f"Distance plot graph throw an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

        finally:
            plt.close('all')

    def plot_violin_graph(self,
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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            if other_feature_name:
                check_if_feature_exists(df,
                                        other_feature_name)

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

            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Violin plot graph couldn't be generated because " +
                    f"there is only missing data to display in {feature_name}!")

            del found_features

            if display_print:
                print("Generating graph violin graph on " + feature_title)

            # Closes up any past graph info
            plt.close('all')

            # Set plot structure
            fig = plt.figure(figsize=figsize)

            plt.title("Violin Plot: " + feature_title)

            feature_values = pd.to_numeric(df[other_feature_name],
                                           errors='coerce').dropna()

            if not len(feature_values):
                raise ValueError("The y feature must contain numerical features.")

            x_values = copy.deepcopy(df[feature_name].dropna())

            # if feature_name in self.__df_features.bool_features():
            #     x_values = pd.to_numeric(x_values,
            #                              errors='ignore')
            #
            #     x_values = ['True' if val == 1 else 'False'
            #                 if val == 0 else val
            #                 for val in x_values]

            # Sort list by x_values
            x_values, feature_values = self.__sort_two_lists(x_values,feature_values)

            warnings.filterwarnings("ignore")

            sns.violinplot(x=x_values,
                           y=feature_values,
                           order=order,
                           cut=cut,
                           scale=scale,
                           gridsize=gridsize,
                           width=width,
                           palette=palette,
                           saturation=saturation)

            warnings.filterwarnings("default")

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
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

            if self.__notebook_mode and display_visuals:
                plt.show()

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            warnings.filterwarnings("default")

            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot violin graph an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

        finally:
            plt.close('all')
            warnings.filterwarnings("default")

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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            # Error check
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

            if feature_name in self.__df_features.bool_features():

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

            plt.title("Category Count Plot: " + feature_name)

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
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

            if self.__notebook_mode and display_visuals:
                plt.show()

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot count graph raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e
        finally:
            plt.close('all')

    def plot_pie_graph(self,
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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

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

            color_list = None

            plt.figure(figsize=figsize)

            # if feature_name in self.__df_features.bool_features():
            #
            #     i = 0
            #     for val in feature_values:
            #         try:
            #             feature_values[i] = float(val)
            #         except:
            #             pass
            #     feature_values = [bool(val) if val == 0 or val == 1 else val
            #                       for val in feature_values]

            # Sort by feature_values
            feature_values,value_count_list = self.__sort_two_lists(feature_values,
                                                                    value_count_list)

            if isinstance(pallete,dict):
                color_list = []
                for value in tuple(feature_values):
                    try:
                        color_list.append(pallete[value])
                    except KeyError:
                        raise KeyError(f"The given value '{value}' in feature '{feature_name}'"
                                       + " was not found in the passed color dict.")

            # Explode the part of the pie graph that is the maximum of the graph
            explode_array = [0] * len(feature_values)
            explode_array[np.array(value_count_list).argmax()] = .03

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
            plt.gcf()
            plt.title("Pie Chart: " + feature_name)
            plt.legend(fancybox=True,
                       facecolor='w')

            # Set foundation
            plt.axis('equal')

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
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)


            if self.__notebook_mode and display_visuals:
                plt.show()

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            if suppress_runtime_errors:
                warnings.warn(
                    f"Pie graph raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e
        finally:
            plt.close('all')

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

        Note -
            A large part of this was taken from: http://tinyurl.com/tuou2cn

        Raises:
           Raises error if the feature data is filled with only nulls or if
           the json file's snapshot of the given dataframe doesn't match the
           given dataframe.
        """
        try:

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            # -----
            check_if_feature_exists(df,
                                    other_feature_name)

            # Error check on null data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Ridge plot graph couldn't be generated because " +
                    f"there is only missing data to display in {feature_name}!")

            if np.sum(df[other_feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Ridge plot graph couldn't be generated because " +
                    f"there is only missing data to display in {other_feature_name}!")

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

            # if feature_name in self.__df_features.bool_features():
            #
            #     tmp_df[feature_name] = pd.to_numeric(tmp_df[feature_name],
            #                                          errors='ignore')
            #     tmp_df[feature_name] = ['True' if val == 1 else 'False'
            #                               if val == 0 else val
            #                             for val in tmp_df[feature_name]]

            tmp_df.dropna(inplace=True)

            # Remove any values that only return a single value back
            for val in tmp_df[feature_name].dropna().unique():

                feature_value_counts = tmp_df[other_feature_name][tmp_df[feature_name] == val].dropna().value_counts()

                count_length = len(feature_value_counts.values)
                if len(feature_value_counts.index.to_list()) <= 1 or count_length == 0:
                    tmp_df = tmp_df[tmp_df[feature_name] != val]

                elif count_length == 1 and feature_value_counts.values[0] == 1:
                    tmp_df = tmp_df[tmp_df[feature_name] != val]

            # -----
            # for val in tmp_df[other_feature_name].dropna().unique():
            #
            #     feature_value_counts = tmp_df[feature_name][tmp_df[other_feature_name] == val].dropna().value_counts()
            #
            #     count_length = len(feature_value_counts.values)
            #     if len(feature_value_counts.index.to_list()) <= 1 or count_length == 0:
            #         tmp_df = tmp_df[tmp_df[other_feature_name] != val]
            #     elif count_length == 1 and feature_value_counts.values[0] == 1:
            #         tmp_df = tmp_df[tmp_df[other_feature_name] != val]

            pd.options.mode.chained_assignment = chained_assignment
            del chained_assignment

            # # Sort by dataframe's series of 'feature_name'
            # tmp_df[feature_name], tmp_df[other_feature_name] = self.__sort_two_lists(tmp_df[feature_name],
            #                                                                          tmp_df[other_feature_name])

            # Suppress any warnings that the seaborn's backend raises
            warnings.filterwarnings("ignore")
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

            g.fig.set_size_inches(figsize[0], figsize[1], forward=True)

            g.fig.suptitle(f'{feature_name} by {other_feature_name}')

            warnings.filterwarnings("default")

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

                if self.__notebook_mode and display_visuals:
                    plt.show()

                self.save_plot(df=df,
                               df_features=self.__df_features,
                               filename=filename,
                               sub_dir=sub_dir,
                               dataframe_snapshot=dataframe_snapshot,
                               suppress_runtime_errors=suppress_runtime_errors,
                               meta_data=not self.__called_from_perform)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            warnings.filterwarnings("default")

            if suppress_runtime_errors:
                warnings.warn(
                    f"Plot ridge graph raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e
        finally:
            plt.close('all')
            warnings.filterwarnings("default")

    def plot_multi_bar_graph(self,
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
                             colors=None,
                             stacked=False):
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

           colors: dict or string
                Dictionary of all feature values to hex color values.

           stacked: bool
                Determines if the multi bar graph should be stacked or not.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """
        try:

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            # -----
            check_if_feature_exists(df,
                                    other_feature_name)


            # Error check on nan data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Multi bar graph on " +
                      f"there is only missing data to display in {feature_name}!")

            if np.sum(df[other_feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Multi bar graph on " +
                                             f"there is only missing data to display in {other_feature_name}!")

            if display_print:
                print(f"Multi bar graph on {feature_name} by {other_feature_name}")

            # Closes up any past graph info
            plt.close('all')


            if not colors:
                try:
                    colors = [self.__df_features.get_feature_colors(feature_name)[val]
                              for val in list(df.groupby(
                            [other_feature_name, feature_name]).size().unstack().columns)]
                except TypeError:
                    pass
                except KeyError:
                    pass

            g = df.groupby([other_feature_name, feature_name]).size().unstack().plot(
                kind='bar',
                stacked=stacked,
                color=colors,
                figsize=figsize)
            g.legend(loc='upper center',
                     bbox_to_anchor=(1.07, 1),
                     shadow=True,
                     ncol=1)
            sns.set(style="whitegrid")

            plt.title(f"Multi bar graph on {feature_name} by {other_feature_name}")


            # Pass a default name if needed
            if not filename:
                if stacked:
                    filename = f"Multi bar graph on {feature_name} by {other_feature_name}"
                else:
                    filename = f"Multi bar graph stacked on {feature_name} by {other_feature_name}"


            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:

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

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            if suppress_runtime_errors:
                warnings.warn(
                    f"Multi bar raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e
        finally:
            plt.close('all')

    def statistical_analysis_on_aggregates(self,
                                           df,
                                           target_features,
                                           dataset_name,
                                           dataframe_snapshot=True):
        """
        Desc:
            Aggregates the data of the target feature either by discrete values
            or by binning/labeling continuous data.

        Args:
           df: pd.Dataframe
               Pandas DataFrame object.

           target_features: list of string
               Specified target features.

           dataset_name: string
               The dataset's name; this will create a sub-directory in which your
               generated graph will be inner-nested in.

           dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

        Note:
            This function has a lot going on and it's infancy so I am going to
            purposely not give it suppress_runtime_errors so people will find
            problems with it and report it to me.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """

        if not self.__called_from_perform and dataframe_snapshot:
            df_snapshot = DataFrameSnapshot()
            df_snapshot.check_create_snapshot(df,
                                              self.__df_features,
                                              directory_path=self.folder_path + f"/{dataset_name}",
                                              sub_dir=f"{dataset_name}/_Extras")

        feature_stats_dict = dict()


        # Convert to list of the given string
        if isinstance(target_features, str):
            target_features = [target_features]

        # Iterate through all target features features
        for feature_name in target_features:

            if feature_name:
                check_if_feature_exists(df,
                                        feature_name)
            else:
                continue

            # Generate bins and labels for continuous numerical data
            bins = None
            labels = None
            if feature_name in self.__df_features.continuous_numerical_features():
                bins, labels = auto_binning(df,
                                            self.__df_features,
                                            feature_name)

            # Labels and bins will act as feature values for aggregation
            if labels:
                feature_values = copy.deepcopy(labels)
            else:
                feature_values = list(df[feature_name].sort_values(
                    ascending=True).dropna().unique())

            # Label to compare to all the data without any aggregations
            feature_values.append("All")

            # Copy feature values to remove one given value from the feature list at a time
            other_feature_values = copy.deepcopy(feature_values)

            # Create target feature dict
            feature_stats_dict[feature_name] = dict()
            feature_val_count = -1
            feature_pvalues = dict()

            # Store all pvalues found for every feature
            for column in df.columns:

                if feature_name == column:
                    continue

                feature_pvalues[column] = dict()

            # Iterate through remaining features
            for main_feature_val in feature_values:

                # Ignore "All" for the main feature val since it doesn't actually exist
                if main_feature_val == "All":
                    continue

                # Don't want repeats or compare same subsets
                other_feature_values.remove(main_feature_val)
                feature_val_count += 1

                # Create series objects based on the main feature val compared to the other_feature_val
                other_feature_val_count = 0
                for other_feature_val in other_feature_values:
                    feature_stats_dict[feature_name][
                        f"{main_feature_val} -> {other_feature_val}"] = dict()

                    # Generate series object based on bins/discrete values
                    other_feature_val_count += 1
                    for iterate_feature_name in df.columns:

                        # -----
                        if iterate_feature_name == feature_name:
                            continue

                        # Create bool array for series object(left)
                        if labels:
                            bool_array = (df[feature_name] <= bins[
                                feature_val_count + 1]) & (
                                                     df[feature_name] > bins[
                                                 feature_val_count])
                        else:
                            bool_array = df[feature_name] == main_feature_val

                        tmp_series_a = df[bool_array][
                            iterate_feature_name].dropna()
                        del bool_array

                        # Create bool array for series object(right)
                        if other_feature_val == "All":
                            bool_array = [True for _ in range(0, df.shape[0])]
                        else:
                            if labels:
                                bool_array = (df[feature_name] <= bins[
                                    feature_val_count + other_feature_val_count + 1]) & (
                                                     df[feature_name] > bins[
                                                 feature_val_count + other_feature_val_count])

                            else:
                                bool_array = df[
                                                 feature_name] == other_feature_val

                        tmp_series_b = df[bool_array][
                            iterate_feature_name].dropna()
                        del bool_array

                        # Extract out pvalue/statistic based on series data
                        if len(tmp_series_a) == 0 or len(tmp_series_b) == 0:
                            pvalue = "NaN"
                            statistic = "NaN"
                        else:
                            ks_2samp = stats.ks_2samp(tmp_series_a,
                                                      tmp_series_b)
                            pvalue = float(ks_2samp.pvalue)
                            statistic = float(ks_2samp.statistic)

                        # Init pvalue/statistic to proper values
                        feature_stats_dict[feature_name][
                            f"{main_feature_val} -> {other_feature_val}"][
                            iterate_feature_name] = dict()
                        feature_stats_dict[feature_name][
                            f"{main_feature_val} -> {other_feature_val}"][
                            iterate_feature_name][
                            "Kolmogorov-Smirnov statistic"] = dict()
                        feature_stats_dict[feature_name][
                            f"{main_feature_val} -> {other_feature_val}"][
                            iterate_feature_name][
                            "Kolmogorov-Smirnov statistic"]["P-Value"] = pvalue

                        feature_stats_dict[feature_name][
                            f"{main_feature_val} -> {other_feature_val}"][
                            iterate_feature_name][
                            "Kolmogorov-Smirnov statistic"][
                            "Statistic"] = statistic

                        # Don't add to list
                        if pvalue == "NaN":
                            continue

                        # Init dict/list if it doesn't exist
                        if "Kolmogorov-Smirnov statistic" not in \
                                feature_pvalues[iterate_feature_name]:
                            feature_pvalues[iterate_feature_name][
                                "Kolmogorov-Smirnov statistic"] = dict()
                            feature_pvalues[iterate_feature_name][
                                "Kolmogorov-Smirnov statistic"][
                                "All pvalues"] = []

                        # Append new pvalue
                        feature_pvalues[iterate_feature_name][
                            "Kolmogorov-Smirnov statistic"][
                            "All pvalues"].append(pvalue)

            # Generate summary data of pvalues
            for column in df.columns:
                if column == feature_name or len(feature_pvalues[column].keys()) == 0:
                    continue

                else:
                    if column in feature_pvalues:
                        feature_pvalues[column][
                            "Kolmogorov-Smirnov statistic"][
                            "All pvalues"].sort()

                        # Only create summary if the series is at least the of 2
                        if len(feature_pvalues[column][
                                   "Kolmogorov-Smirnov statistic"][
                                   "All pvalues"]) >= 2:

                            feature_pvalues[column][
                                "Kolmogorov-Smirnov statistic"
                            ]["Pvalues Summary"] = descr_table(
                                pd.DataFrame({column: feature_pvalues[column][
                                    "Kolmogorov-Smirnov statistic"][
                                    "All pvalues"]}),
                                column).to_dict()[column]
                        # Init to an empty dict
                        else:
                            feature_pvalues[column][
                                "Kolmogorov-Smirnov statistic"][
                                "Pvalues Summary"] = {}

            feature_stats_dict[feature_name]["P-Values"] = feature_pvalues
        # End target feature loop

        # Generate directories
        create_dir_structure(self.folder_path + dataset_name,
                             "_Extras/Statistics/Accept Null Hypothesis")
        create_dir_structure(self.folder_path + dataset_name,
                             "_Extras/Statistics/Reject Null Hypothesis")

        # Create json file
        dict_to_json_file(feature_stats_dict,
                          self.folder_path + dataset_name + "/_Extras/Statistics",
                          "Statistics on target features")

        stat_methods_dict = dict()

        for main_feature, relationship_dict in feature_stats_dict.items():
            for _, stats_on_features in relationship_dict.items():
                for iterate_feature_name, stats_method_dict in stats_on_features.items():
                    for stats_method, stats_dict in stats_method_dict.items():
                        if "All pvalues" in stats_dict:

                            if stats_method not in stat_methods_dict:
                                stat_methods_dict[stats_method] = pd.DataFrame()

                            stats_dict = copy.deepcopy(stats_dict)

                            for k,v in stats_dict.items():
                                if v == "NaN":
                                    stats_dict[k] = [np.nan]
                                else:
                                    stats_dict[k] = [v]

                            if len(stats_dict['Pvalues Summary'][0]) > 0:
                                tmp_stats_df = pd.DataFrame.from_dict(stats_dict["Pvalues Summary"])[["mean","std","var"]]
                                tmp_stats_df.index = [f"{main_feature} compared to {iterate_feature_name}"]
                                stat_methods_dict[stats_method] = stat_methods_dict[stats_method].append(tmp_stats_df,
                                                                                           ignore_index=False)
        for stats_method in stat_methods_dict:
            if stat_methods_dict[stats_method].shape[0]:
                stat_methods_dict[stats_method].sort_values(
                    by=["mean", "std", "var"],
                    ascending=True,
                    inplace=True)

        pickle_object_to_file(stat_methods_dict,
                              self.folder_path + dataset_name + "/_Extras/Statistics",
                              "Stat methods of features dataframes",
                              remove_file_extension=False)

        # Generate multiple json files based on the following pvalues
        for accept_null_plvalue in [.01, .05, .1, .101, .2, .3, .4, .5, .6, .7,
                                    .8, .9, 1]:
            json_dict = copy.deepcopy(feature_stats_dict)
            tmp_dict = copy.deepcopy(feature_stats_dict)
            for main_feature, relationship_dict in tmp_dict.items():
                for relationship_string, stats_on_features in relationship_dict.items():
                    for iterate_feature_name, stats_method_dict in stats_on_features.items():
                        for stats_method, stats_dict in stats_method_dict.items():

                            # Not a relationship string; Re-access pvalue list and summary
                            if relationship_string == "P-Values":

                                if "P-Values" not in json_dict[
                                    main_feature] or iterate_feature_name not in \
                                        json_dict[main_feature]["P-Values"]:
                                    break

                                filter_pvalues = np.asarray(
                                    json_dict[main_feature]["P-Values"][
                                        iterate_feature_name][stats_method][
                                        "All pvalues"])

                                if accept_null_plvalue <= .1:
                                    filter_pvalues = filter_pvalues[
                                        filter_pvalues <= accept_null_plvalue]
                                else:
                                    filter_pvalues = filter_pvalues[
                                        filter_pvalues >= accept_null_plvalue]

                                json_dict[main_feature]["P-Values"][
                                    iterate_feature_name][stats_method][
                                    "All pvalues"] = list(filter_pvalues)

                                if len(filter_pvalues) >= 2:
                                    json_dict[main_feature][
                                        "P-Values"][iterate_feature_name][
                                        stats_method]["Pvalues Summary"] = \
                                    descr_table(
                                        pd.DataFrame(
                                            {iterate_feature_name: list(
                                                filter_pvalues)}),
                                        iterate_feature_name).to_dict()[
                                        iterate_feature_name]
                                else:
                                    json_dict[main_feature][
                                        "P-Values"][iterate_feature_name][
                                        stats_method]["Pvalues Summary"] = {}

                                break

                            pvalue = stats_dict["P-Value"]
                            if accept_null_plvalue <= .1:
                                if pvalue == "NaN" or pvalue > accept_null_plvalue:
                                    del json_dict[main_feature][
                                        relationship_string][
                                        iterate_feature_name]
                            else:
                                if pvalue == "NaN" or pvalue < accept_null_plvalue:
                                    del json_dict[main_feature][
                                        relationship_string][
                                        iterate_feature_name]

            # Push to accept or reject null hypothesis folder
            if accept_null_plvalue <= .1:
                dict_to_json_file(json_dict,
                                  self.folder_path + dataset_name + "/_Extras/Statistics/Accept Null Hypothesis",
                                  f"Accept Null Hypothesis on target features where pvalue <= {accept_null_plvalue}",
                                  remove_file_extension=False)
            else:
                dict_to_json_file(json_dict,
                                  self.folder_path + dataset_name + "/_Extras/Statistics/Reject Null Hypothesis",
                                  f"Reject Null Hypothesis on target features where pvalue >= {accept_null_plvalue}",
                                  remove_file_extension=False)

    def plot_jointplot_graph(self,
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
                             color=None,
                             kind="scatter and kde",
                             ratio=5):

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
                Tuple object to represent the plot/image's size. Because joinplot
                only accepts a single value for the figure; we just pull the
                greatest of the two values.

            color: string
                Seaborn/maplotlib color/hex color for representing the graph

            kind: string (scatter,reg,resid,kde,hex,scatter and kde)
                Kind of plot to draw.

            ratio:
                Ratio of joint axes height to marginal axes height.
                (Determines distplot like plots dimensions.)

            Credit to seaborn's author:
            Michael Waskom
            Git username: mwaskom
            Link: http://tinyurl.com/v9pxsoy

        Raises:
           Raises error if the feature data is filled with only nulls or if
           the json file's snapshot of the given dataframe doesn't match the
           given dataframe.
        """


        try:

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            check_if_feature_exists(df,
                                    other_feature_name)

            # Error check on null data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Jointplot plot graph couldn't be generated because " +
                    f"there is only missing data to display in {feature_name}!")

            if np.sum(df[other_feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments(
                    "Jointplot plot graph couldn't be generated because " +
                    f"there is only missing data to display in {other_feature_name}!")

            if display_print:
                print(f"Generating jointplot graph on {feature_name} by {other_feature_name}")

            # Closes up any past graph info
            plt.close('all')

            if figsize[0] < figsize[1]:
                height = figsize[0]
            else:
                height = figsize[1]

            tmp_df = copy.deepcopy(df[[feature_name,other_feature_name]])
            tmp_df.dropna()

            if not kind:
                kind = "scatter"

            warnings.filterwarnings("ignore")
            if kind == "scatter and kde":
                g = sns.jointplot(feature_name,
                                  other_feature_name,
                                  data=tmp_df,
                                  kind="scatter",
                                  color=color,
                                  ratio=ratio,
                                  height=height).plot_joint(sns.kdeplot, zorder=0,
                                                            n_levels=6)
            else:
                g = sns.jointplot(feature_name,
                                  other_feature_name,
                                  data=tmp_df,
                                  kind=kind,
                                  color=color,
                                  ratio=ratio,
                                  height=height)

            warnings.filterwarnings("default")

            plt.subplots_adjust(top=0.93)
            g.fig.suptitle("Jointplot: " + f"{feature_name} by {other_feature_name}")

            # Pass a default name if needed
            if not filename:
                filename = f"Jointplot plot graph for {feature_name} by {other_feature_name} using {kind}"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            # -----
            if save_file:

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

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:
            warnings.filterwarnings("default")

            if suppress_runtime_errors:
                warnings.warn(
                    f"Joinplot plot graph an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

        finally:
            plt.close('all')
            warnings.filterwarnings("default")


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
        Desc:
            Creates a value counts table of the features given data.

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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

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
                                        df_features=self.__df_features,
                                        filename=filename,
                                        sub_dir=sub_dir,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        table=val_counts_df,
                                        show_index=True,
                                        meta_data=not self.__called_from_perform)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            if suppress_runtime_errors:
                warnings.warn(
                    f"Value count table raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e
        finally:
            plt.close('all')


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
            Creates/Saves a pandas dataframe of a feature's numerical data.
            Standard deviation, mean, Q1-Q5, median, variance, etc.

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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            # Check if dataframe has only null data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Descr table couldn't be generated because " +
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
                                        df_features=self.__df_features,
                                        filename=filename,
                                        sub_dir=sub_dir,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        table=desc_df,
                                        meta_data=not self.__called_from_perform,
                                        show_index=True)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            if suppress_runtime_errors:
                warnings.warn(
                    f"Descr table raised an error on feature '{feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

        finally:
            plt.close('all')

    def group_by_feature_value_count_table(self,
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

            # -----
            check_if_feature_exists(df,
                                    feature_name)

            # Check if dataframe has only null data
            if np.sum(df[feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                      f"there is only missing data to display in {feature_name}!")

            if np.sum(df[other_feature_name].isnull()) == df.shape[0]:
                raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                      f"there is only missing data to display in {other_feature_name}!")

            if display_print:
                print(f"Creating group by {feature_name} and {other_feature_name} Table")

            tmp_df = copy.deepcopy(df[[feature_name, other_feature_name]])
            tmp_df = tmp_df.groupby([feature_name, other_feature_name]).size().to_frame()
            tmp_df.columns = ["Counts"]

            if self.__notebook_mode:
                if display_visuals:
                    display(tmp_df)
            else:
                if display_visuals:
                    print(tmp_df)

            # Pass a default name if needed
            if not filename:
                filename = f"Group by {feature_name} and {other_feature_name} Table"

            # Create string sub directory path
            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"

            tmp_df.sort_values(by=["Counts"],
                               ascending=False,
                               inplace=True)

            if save_file:

                if self.__called_from_perform:
                    dataframe_snapshot = False

                self.save_table_as_plot(df=tmp_df,
                                        df_features=self.__df_features,
                                        filename=filename,
                                        sub_dir=sub_dir,
                                        dataframe_snapshot=dataframe_snapshot,
                                        suppress_runtime_errors=suppress_runtime_errors,
                                        table=tmp_df,
                                        show_index=True,
                                        meta_data=not self.__called_from_perform)

        except SnapshotMismatchError as e:
            raise e

        except Exception as e:

            if suppress_runtime_errors:
                warnings.warn(
                    f"Group by table raised an error on feature '{feature_name}' by '{other_feature_name}':\n{str(e)}",
                    RuntimeWarning)
            else:
                raise e

        finally:
            plt.close('all')


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

    def __sort_two_lists(self,
                         sort_values,
                         other_list):
        """
        Desc:
            Sort's two collections by the first collection passed in.

        Args:
            sort_values: collection
                Values to be sorted by.
            other_list: collection
                Values that get sorted based on 'sort_values'.
        Returns:
            Returns back those two lists sorted.
        """
        tmp = list(zip(*sorted(list(zip(other_list, sort_values)),
                               key=lambda x: x[1])))

        return list(tmp[1]), list(tmp[0])