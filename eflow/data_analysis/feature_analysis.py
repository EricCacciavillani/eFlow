from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow.utils.pandas_utils import descr_table,value_counts_table, df_to_image
from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import convert_to_filename
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow._hidden.custom_warnings import EflowWarning
from eflow._hidden.constants import GRAPH_DEFAULTS


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


class FeatureAnalysis(FileOutput):

    """
        Analyzes the feature data of a pandas Dataframe object.
        (Only works on single features and ignores null data for displaying data.)
    """

    def __init__(self,
                 df_features,
                 project_sub_dir="",
                 project_name="Data Analysis",
                 overwrite_full_path=None,
                 notebook_mode=True):
        """
        Args:
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
                         dataframe_snapshot=True):
        """
        Desc:
            Performs all public methods that generate visualizations/insights
            about the data.

        Note:
            Pretty much my personal lazy button for running the entire object
            without specifying any method in particular.

        Args:
            df: Pandas dataframe
                Pandas dataframe object

            df_features: DataFrameTypes
                DataFrameTypes object; organizes feature types into groups.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

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

            target_feature = self.__df_features.get_target_feature()
            # Iterate through features
            for feature_name in df.columns:

               try:
                   self.generate_graphics_for_feature(df,
                                                      feature_name,
                                                      dataset_name,
                                                      display_visuals=display_visuals,
                                                      save_file=save_file,
                                                      dataframe_snapshot=dataframe_snapshot)
               except ValueError as e:
                   print(f"Error found on feature {feature_name}: {e}")

               if feature_name == target_feature:

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

                           for f_name in df.columns:

                               if f_name == target_feature:
                                   continue

                               if repr_target_feature_val:
                                    print(f"Target feature {target_feature} set to {target_feature_val}; also known as {repr_target_feature_val}.")
                               else:
                                   print(f"Target feature {target_feature} set to {target_feature_val}.")
                               try:
                                   self.generate_graphics_for_feature(df[df[target_feature] == target_feature_val],
                                                                      f_name,
                                                                      dataset_name,
                                                                      display_visuals=display_visuals,
                                                                      save_file=save_file,
                                                                      dataframe_snapshot=dataframe_snapshot,
                                                                      sub_dir=f"{dataset_name}/{target_feature}/{repr_target_feature_val}/{f_name}")
                               except Exception as e:
                                   print(f"Error found on feature {f_name}: {e}")

            missed_features = set(df.columns) ^ self.__df_features.get_all_features()
            # If any missed features are picked up...
            if len(missed_features) != 0:
                print("Some features were not analyzed by perform analysis!")
                for feature_name in missed_features:
                    print(f"\t\tFeature:{feature_name}")

        # Ensures that called from perform is turned off
        finally:
            self.__called_from_perform = False

    def plot_distance_graph(self,
                            df,
                            feature_name,
                            dataset_name,
                            display_visuals=True,
                            filename=None,
                            sub_dir=None,
                            save_file=True,
                            dataframe_snapshot=True,
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
            df: Pandas dataframe
                Pandas dataframe object

            feature_name: string
                Specified feature column name.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            filename: string
                Name to give the file.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

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

            color : matplotlib color
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
        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments(
                "Distance plot graph couldn't be generated because " +
                f"there is only missing data to display in {feature_name}!")

        print(f"Generating graph for distance plot on {feature_name}")

        feature_values = pd.to_numeric(df[feature_name].dropna(),
                                       errors='coerce').dropna()

        if not len(feature_values):
            raise ValueError(
                f"The given feature {feature_name} doesn't seem to convert to a numeric vector.")

        # Closes up any past graph info
        plt.close()

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

        # -----
        if save_file:

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir, str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Check if dataframe matches saved snapshot; Creates file if needed
            if not self.__called_from_perform:

                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{sub_dir}/_Extras")

            # Create the png
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))


        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()


    def plot_violin_graph(self,
                          df,
                          feature_name,
                          dataset_name,
                          y_feature_name=None,
                          display_visuals=True,
                          filename=None,
                          sub_dir=None,
                          save_file=True,
                          dataframe_snapshot=True,
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
            df: Pandas dataframe
                Pandas dataframe object

            feature_name: string
                Specified feature column name to compare to y.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            y_feature_name: string
                Specified feature column name to compare to x.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            filename: string
                Name to give the file.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

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

            colors: matplotlib color, optional
                Color for all of the elements, or seed for a gradient palette.

            palette:
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

        # Error check and create title/part of default file name
        found_features = []
        feature_title = ""
        for feature in (feature_name, y_feature_name):
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

        print("Generating graph violin graph on " + feature_title)

        # Closes up any past graph info
        plt.close()

        # Set plot structure
        plt.figure(figsize=figsize)
        plt.title("Violin Plot: " + feature_title)

        feature_values = pd.to_numeric(df[y_feature_name],
                                       errors='coerce').dropna()

        if not len(feature_values):
            raise ValueError("The y feature must contain numerical features.")

        x_values = df[feature_name]
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


        # -----
        if save_file:

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir, str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Check if dataframe matches saved snapshot; Creates file if needed
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{sub_dir}/_Extras")

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir, str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Create the png
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))


        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()

    def plot_count_graph(self,
                         df,
                         feature_name,
                         dataset_name,
                         display_visuals=True,
                         filename=None,
                         sub_dir=None,
                         save_file=True,
                         dataframe_snapshot=True,
                         figsize=GRAPH_DEFAULTS.FIGSIZE,
                         flip_axis=False,
                         palette="PuBu"):
        """
        Desc:
            Display a barplot with color ranking from a feature's value counts
            from the seaborn libary and save the graph in the correct directory
            structure.

        Args:
            df: Pandas dataframe
                Pandas dataframe object.

            feature_name: string
                Specified feature column name.

            dataset_name: string
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            filename: string
                Name to give the file.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            figsize: tuple
                Size for the given plot.

            flip_axis: bool
                Flip the axis the ploting axis from x to y if set to 'True'.

            palette: string
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

        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments(
                "Count plot graph couldn't be generated because " +
                f"there is only missing data to display in {feature_name}!")
        print(f"Count plot graph on {feature_name}")

        # Closes up any past graph info
        plt.close()

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
        # -----
        if save_file:

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir, str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Check if dataframe matches saved snapshot; Creates file if needed
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{sub_dir}/_Extras")

            # Create the png
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()

    def pie_graph(self,
                  df,
                  feature_name,
                  dataset_name,
                  display_visuals=True,
                  filename=None,
                  sub_dir=None,
                  save_file=True,
                  dataframe_snapshot=True,
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

           colors:
                Dictionary of all feature values to hex color values.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """

        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments("Pie graph couldn't be generated because " +
                  f"there is only missing data to display in {feature_name}!")
        print(f"Pie graph on {feature_name}")

        # Closes up any past graph info
        plt.close()

        # Find value counts
        value_counts = df[feature_name].dropna().value_counts(sort=False)
        value_list = value_counts.index.tolist()
        value_count_list = value_counts.values.tolist()

        # Explode the part of the pie graph that is the maximum of the graph
        explode_array = [0] * len(value_list)
        explode_array[np.array(value_count_list).argmax()] = .03

        color_list = None

        if isinstance(pallete,dict):
            color_list = []
            for value in tuple(value_list):
                try:
                    color_list.append(pallete[value])
                except KeyError:
                    raise KeyError(f"The given value '{value}' in feature '{feature_name}'"
                                   + " was not found in the passed color dict.")

        plt.figure(figsize=figsize)

        if feature_name in self.__df_features.get_bool_features():
            value_list = [bool(val) if val == 0 or val == 1 else val
                          for val in value_list]

        # Plot pie graph
        plt.pie(
            tuple(value_count_list),
            labels=tuple(value_list),
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

        # -----
        if save_file:

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir, str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            # Check if dataframe matches saved snapshot; Creates file if needed
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{sub_dir}/_Extras")

            # Create the png
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))


        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()


    def value_counts_table(self,
                           df,
                           feature_name,
                           dataset_name,
                           display_visuals=True,
                           filename=None,
                           sub_dir=None,
                           save_file=True,
                           dataframe_snapshot=True):
        """
        Args:
            df:
                Pandas DataFrame object

            feature_name:
                Specified feature column name.

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            save_file:
                Saves file if set to True; doesn't if set to False.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

        Desc:
            Creates/Saves a pandas dataframe of value counts of a dataframe.

            Note:
                Creates a png of the table.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """

        # Check if feature has only null data
        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments("Values count table couldn't be generated because " +
                                         f"there is only missing data to display in {feature_name}!")

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

        if save_file:

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir,str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{sub_dir}/_Extras")
            # Closes up any past graph info
            plt.close()

            # Convert value counts dataframe to an image
            df_to_image(val_counts_df,
                        self.folder_path,
                        sub_dir,
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)

    def descr_table(self,
                    df,
                    feature_name,
                    dataset_name,
                    display_visuals=True,
                    filename=None,
                    sub_dir=None,
                    save_file=True,
                    dataframe_snapshot=True):
        """
        Desc:
            Creates/Saves a pandas dataframe of features and their found types
            in the dataframe.

            Note:
                Creates a png of the table.

        Args:
            df:
                Pandas DataFrame object

            feature_name:
                Specified feature column name.

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            display_visuals:
                Boolean value to whether or not to display visualizations.

            filename:
                If set to 'None' will default to a pre-defined string;
                unless it is set to an actual filename.

            sub_dir:
                Specify the sub directory to append to the pre-defined folder path.

            save_file:
                Saves file if set to True; doesn't if set to False.

            dataframe_snapshot:
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

        Raises:
            Raises error if the feature data is filled with only nulls or if
            the json file's snapshot of the given dataframe doesn't match the
            given dataframe.
        """

        # Check if dataframe has only null data
        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                  f"there is only missing data to display in {feature_name}!")

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

        if save_file:

            if not sub_dir:
                sub_dir = f"{dataset_name}/{feature_name}"
            if not isinstance(sub_dir,str):
                raise TypeError(f"Expected param 'sub_dir' to be type string! Was found to be {type(sub_dir)}")

            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      self.__df_features,
                                                      directory_path=self.folder_path,
                                                      sub_dir=f"{sub_dir}/_Extras")
            # Closes up any past graph info
            plt.close()

            # Convert value counts dataframe to an image
            df_to_image(desc_df,
                        self.folder_path,
                        sub_dir,
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)



    def generate_graphics_for_feature(self,
                                      df,
                                      feature_name,
                                      dataset_name,
                                      display_visuals=True,
                                      sub_dir=None,
                                      save_file=True,
                                      dataframe_snapshot=True):

        colors = self.__get_feature_colors(df,
                                           feature_name)

        # Display colors
        if colors:
            print(f"Colors:\n{colors}\n")

        target_feature = self.__df_features.get_target_feature()
        target_feature_cont_numerical = target_feature in self.__df_features.get_continuous_numerical_features()

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
                               dataframe_snapshot=dataframe_snapshot)

            self.plot_count_graph(df,
                                  feature_name,
                                  dataset_name=dataset_name,
                                  display_visuals=display_visuals,
                                  sub_dir=sub_dir,
                                  save_file=save_file,
                                  palette=colors,
                                  dataframe_snapshot=dataframe_snapshot)

            if colors:
                self.plot_count_graph(df,
                                      feature_name,
                                      dataset_name=dataset_name,
                                      display_visuals=display_visuals,
                                      sub_dir=sub_dir,
                                      save_file=save_file,
                                      dataframe_snapshot=dataframe_snapshot)

            self.value_counts_table(df,
                                    feature_name,
                                    dataset_name=dataset_name,
                                    display_visuals=display_visuals,
                                    sub_dir=sub_dir,
                                    save_file=save_file,
                                    dataframe_snapshot=dataframe_snapshot)

            if target_feature and feature_name != target_feature:

                if target_feature_cont_numerical:
                    self.plot_violin_graph(df,
                                           feature_name,
                                           dataset_name=dataset_name,
                                           y_feature_name=target_feature,
                                           display_visuals=display_visuals,
                                           sub_dir=sub_dir,
                                           save_file=save_file,
                                           dataframe_snapshot=dataframe_snapshot)
                else:
                    pass

            print("\n\n")

        # -----
        elif feature_name in self.__df_features.get_continuous_numerical_features():
            self.plot_distance_graph(df,
                                     feature_name,
                                     dataset_name=dataset_name,
                                     display_visuals=display_visuals,
                                     sub_dir=sub_dir,
                                     save_file=save_file,
                                     dataframe_snapshot=dataframe_snapshot)

            self.descr_table(df,
                             feature_name,
                             dataset_name=dataset_name,
                             display_visuals=display_visuals,
                             sub_dir=sub_dir,
                             save_file=save_file,
                             dataframe_snapshot=dataframe_snapshot)

            if target_feature and feature_name != target_feature:

                if target_feature_cont_numerical:
                    pass
                else:
                    self.plot_violin_graph(df,
                                           target_feature,
                                           dataset_name=dataset_name,
                                           y_feature_name=feature_name,
                                           display_visuals=display_visuals,
                                           sub_dir=sub_dir,
                                           save_file=save_file,
                                           dataframe_snapshot=dataframe_snapshot)

    def __get_feature_colors(self,
                             df,
                             feature_name):
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