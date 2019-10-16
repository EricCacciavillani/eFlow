import random
import numpy as np
from matplotlib import pyplot as plt
import copy
from IPython.display import display
import seaborn as sns

from eflow.utils.sys_utils import *
from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.custom_exceptions import *

from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.general_objects import DataFrameSnapshot
from eflow.utils.pandas_utils import descr_table,value_counts_table
from eflow.utils.image_utils import create_plt_png, df_to_image
from eflow.utils.string_utils import convert_to_filename
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments


class FeatureAnalysis(FileOutput):

    """
        Analyzes the feature data of a pandas Dataframe object.
        (Only works on single features and ignores null data for displaying data.)
    """

    def __init__(self,
                 project_sub_dir="",
                 project_name="Data Analysis",
                 overwrite_full_path=None,
                 notebook_mode=True):
        """
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

        # Pre-defined colors for column's with set column names names.
        # Multiple names/values are allowed
        self.__defined_column_colors = list()
        self.__defined_column_colors.append([["gender", "sex"],
                                             ["Male", "M", "#7EAED3"],
                                             ["Female", "F", "#FFB6C1"]])
        self.__defined_column_colors.append([[" "],
                                             ["Male", "#7EAED3"],
                                             ["Female", "#FFB6C1"]])
        self.__defined_column_colors.append([[" "],
                                             ["Y", "y" "yes", "Yes",
                                              "#55a868"],
                                             ["N", "n", "no", "No",
                                              "#ff8585"]])
        self.__defined_column_colors.append([[" "],
                                             [True, "True", "#55a868"],
                                             [False, "False", "#ff8585"]])

        self.__notebook_mode = copy.deepcopy(notebook_mode)
        self.__called_from_perform = False

    def __check_specfied_column_colors(self,
                                       df,
                                       feature_name,
                                       init_default_color=None):
        """
        df:
            Pandas DataFrame object.

        col_feature_name:
            Specified feature column name.

        init_default_color:
            A default color to assign unknown values when other values are
            already assigned. Left to 'None' will init with random colors.

        Returns/Descr:
            Checks the column name and assigns it with the appropriate
            color values if the values also match specified values.
        """

        specfied_column_values = [
            str(x).upper() for x in
            df[feature_name].value_counts().index.tolist()]

        # Assign with default color value or assign random colors
        if not init_default_color:
            column_colors = ["#%06x" % random.randint(0, 0xFFFFFF)
                             for _ in range(0,
                                            len(specfied_column_values))]
        else:
            column_colors = [init_default_color
                             for _ in range(0,
                                            len(
                                                specfied_column_values))]

        found_color_value = False

        # Check if the given column name matches any pre-defined names
        for column_info in copy.deepcopy(self.__defined_column_colors):

            specified_column_names = column_info.pop(0)
            specified_column_names = [str(x).upper()
                                      for x in specified_column_names]

            # Compare both feature names; ignore char case; check for default
            if feature_name.upper() in specified_column_names or \
                    specified_column_names[0] == " ":

                for column_value_info in column_info:
                    column_value_color = column_value_info.pop(-1)

                    for column_value in {x for x in column_value_info}:

                        if str(column_value).upper() in specfied_column_values:
                            column_colors[specfied_column_values.index(str(
                                column_value).upper())] = column_value_color
                            found_color_value = True

                # No colors were found reloop operation
                if not found_color_value:
                    continue
                else:
                    return column_colors

        # Return obj None for no matching colors
        return None

    def perform_analysis(self,
                         df,
                         df_features,
                         dataset_name,
                         display_visuals=True,
                         save_file=True,
                         dataframe_snapshot=True):
        """
        df:
            Pandas dataframe object

        df_features:
            DataFrameTypes object; organizes feature types into groups.

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
        """
        try:
           self.__called_from_perform = False
           if df.shape[0] == 0 or np.sum(np.sum(df.isnull()).values) == df.shape[0]:
               raise UnsatisfiedRequirments("Dataframe must contain valid data and not be empty or filled with nulls!")
           # Iterate through DataFrame columns and graph based on data types

           if dataframe_snapshot:
               df_snapshot = DataFrameSnapshot()
               df_snapshot.check_create_snapshot(df,
                                                 directory_pth=self.folder_path,
                                                 sub_dir=f"{dataset_name}/_Extras")

           for feature_name in df.columns:

               feature_values = df[feature_name].value_counts().keys()
               if len(feature_values) <= 3 and \
                       not feature_name in df_features.get_numerical_features():
                   self.pie_graph(df,
                                  feature_name,
                                  dataset_name=dataset_name,
                                  display_visuals=display_visuals,
                                  save_file=save_file,
                                  init_default_color="#C0C0C0")

               elif feature_name in df_features.get_categorical_features():
                   self.count_plot_graph(df,
                                         feature_name,
                                         dataset_name=dataset_name,
                                         display_visuals=display_visuals,
                                         save_file=save_file)

               elif feature_name in df_features.get_integer_features():
                   if len(feature_values) <= 13:
                       self.count_plot_graph(df,
                                             feature_name,
                                             dataset_name=dataset_name,
                                             display_visuals=display_visuals,
                                             save_file=save_file)
                   else:
                       self.distance_plot_graph(df,
                                                feature_name,
                                                dataset_name=dataset_name,
                                                display_visuals=display_visuals,
                                                save_file=save_file)

               elif feature_name in df_features.get_float_features():
                   self.distance_plot_graph(df,
                                            feature_name,
                                            dataset_name=dataset_name,
                                            display_visuals=display_visuals,
                                            save_file=save_file)

           else:
               print(
                   "Object didn't receive a Pandas Dataframe object or a DataFrameTypes object")
        finally:
           self.__called_from_perform = False

    def distance_plot_graph(self,
                            df,
                            feature_name,
                            dataset_name,
                            display_visuals=True,
                            filename=None,
                            save_file=True,
                            dataframe_snapshot=True):
        """
        df:
            Pandas dataframe object

        feature_name:
            Specified feature column name.

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

        Returns/Descr:
            Display a distance plot and save the graph/table in the correct
            directory.
        """

        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments(
                "Distance plot graph couldn't be generated because " +
                f"there is only missing data to display in {feature_name}!")

        print(f"Generating graph for distance plot graph on {feature_name}")
        plt.close()

        # Set foundation graph info
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        plt.title("Distance Plot: " + feature_name)

        # Create seaborn graph
        sns.distplot(df[feature_name].dropna())

        if not filename:
            filename = f"Distance plot graph on {feature_name}"

        if save_file:

            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
                           convert_to_filename(filename))


        if self.__notebook_mode and display_visuals:
            plt.show()

        plt.close()


    def count_plot_graph(self,
                         df,
                         feature_name,
                         dataset_name,
                         display_visuals=True,
                         filename=None,
                         save_file=True,
                         dataframe_snapshot=True,
                         flip_axis=False,
                         palette="PuBu"):
        """
        df:
            Pandas dataframe object

        feature_name:
            Specified feature column name.

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


        flip_axis:
            Flip the x and y axis for visual representation.

        palette:
            Seaborn color palette, specifies the colors the graph will use.

        Returns/Descr:
            Display a count plot and save the graph/table in the correct
            directory.
        """

        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                  f"there is only missing data to display in {feature_name}!")
        print(
            f"Count plot graph for distance plot graph on {feature_name}")
        plt.close()

        # Set graph info
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        plt.title("Category Count Plot: " + feature_name)

        # Find and rank values based on counts for color variation of the graph
        groupedvalues = df.groupby(feature_name).sum().reset_index()

        pal = sns.color_palette(palette, len(groupedvalues))

        rank_list = []
        for target_value in df[feature_name].dropna().unique():
            rank_list.append(sum(
                df[feature_name] == target_value))

        rank_list = np.argsort(-np.array(rank_list)).argsort()

        # Flip the graph for visual flare
        if flip_axis:
            ax = sns.countplot(y=feature_name, data=df,
                               palette=np.array(pal[::-1])[rank_list])
        else:
            ax = sns.countplot(x=feature_name, data=df,
                               palette=np.array(pal[::-1])[rank_list])

        # Labels for numerical count of each bar
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1}'.format(height),
                    ha="center")

        if not filename:
            filename = f"Count plot graph on {feature_name}"

        if save_file:

            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
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
                  save_file=True,
                  dataframe_snapshot=True,
                  colors=None,
                  init_default_color=None):
        """
       df:
           Pandas DataFrame object.

       feature_name:
           Specified feature column name.

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

       colors:
            Accepts an array of hex colors with the correct count of values
            within the feature. If not init; then specified colors will be
            assigned based on if the feature is Boolean or if the column name
            is found in 'defined_column_colors'; else just init with
            random colors.

       init_default_color:
           A default color to assign unknown values when other values are
           already assigned. Left to 'None' will init with random colors.

       Returns/Descr:
           Display a pie graph and save the graph/table in the correct
           directory.
        """

        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments("Pie graph couldn't be generated because " +
                  f"there is only missing data to display in {feature_name}!")
        print(
            f"Pie graph for distance plot graph on {feature_name}")
        plt.close()

        # Find value counts
        value_counts = df[feature_name].dropna().value_counts()
        value_list = value_counts.index.tolist()
        value_count_list = value_counts.values.tolist()

        # Init with proper color hex value based on conditionals. (Read Above)
        if colors is None:
            colors = self.__check_specfied_column_colors(df,
                                                         feature_name,
                                                         init_default_color)

        # Explode the part of the pie graph that is the maximum of the graph
        explode_array = [0] * len(value_list)
        explode_array[np.array(value_count_list).argmax()] = .03

        # Plot pie graph
        plt.pie(
            tuple(value_count_list),
            labels=tuple(value_list),
            shadow=False,
            colors=colors,
            explode=tuple(explode_array),
            startangle=90,
            autopct='%1.1f%%',
        )

        # Set foundation graph info
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.title("Pie Chart: " + feature_name)
        plt.legend(fancybox=True)
        plt.axis('equal')
        plt.tight_layout()
        plt.figure(figsize=(20, 20))

        if not filename:
            filename = f"Pie graph on {feature_name}"

        if save_file:

            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")

            create_plt_png(self.folder_path,
                           f"{dataset_name}/Graphics",
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
                           save_file=True,
                           dataframe_snapshot=True):
        """
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

        save_file:
            Saves file if set to True; doesn't if set to False.

        dataframe_snapshot:
            Boolean value to determine whether or not generate and compare a
            snapshot of the dataframe in the dataset's directory structure.
            Helps ensure that data generated in that directory is correctly
            associated to a dataframe.

        Returns/Desc:
            Creates/Saves a pandas dataframe of value counts of a dataframe.
        """

        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            raise UnsatisfiedRequirments("Values count table couldn't be generated because " +
                                         f"there is only missing data to display in {feature_name}!")

        print("Creating data description table...")

        val_counts_df = value_counts_table(df,
                                           feature_name)

        if self.__notebook_mode:
            if display_visuals:
                display(val_counts_df)
        else:
            if display_visuals:
                print(val_counts_df)

        if not filename:
            filename = f"{feature_name} Value Counts Table"

        if save_file:
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")
            plt.close()
            df_to_image(val_counts_df,
                        self.folder_path,
                        f"{dataset_name}/Tables",
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)

    def descr_table(self,
                    df,
                    feature_name,
                    dataset_name,
                    display_visuals=True,
                    filename=None,
                    save_file=True,
                    dataframe_snapshot=True):
        """
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

        save_file:
            Saves file if set to True; doesn't if set to False.

        dataframe_snapshot:
            Boolean value to determine whether or not generate and compare a
            snapshot of the dataframe in the dataset's directory structure.
            Helps ensure that data generated in that directory is correctly
            associated to a dataframe.

        Returns/Desc:
            Creates/Saves a pandas dataframe of features and their found types
            in the dataframe.
        """
        if np.sum(df[feature_name].isnull()) == df.shape[0]:
            print("This function requires a dataframe"
                  "in both rows and columns.")
            raise UnsatisfiedRequirments("Count plot graph couldn't be generated because " +
                  f"there is only missing data to display in {feature_name}!")
            return None

        print("Creating data description table...")

        col_desc_df = descr_table(df,
                                  feature_name)

        if self.__notebook_mode:
            if display_visuals:
                display(col_desc_df)
        else:
            if display_visuals:
                print(col_desc_df)

        if not filename:
            filename = f"{feature_name} Description Table"

        if save_file:
            if not self.__called_from_perform:
                if dataframe_snapshot:
                    df_snapshot = DataFrameSnapshot()
                    df_snapshot.check_create_snapshot(df,
                                                      directory_pth=self.folder_path,
                                                      sub_dir=f"{dataset_name}/_Extras")
            plt.close()
            df_to_image(col_desc_df,
                        self.folder_path,
                        f"{dataset_name}/Tables",
                        convert_to_filename(filename),
                        show_index=True,
                        format_float_pos=2)