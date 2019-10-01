import random
import numpy as np
from matplotlib import pyplot as plt
import copy
from IPython.display import display
import seaborn as sns

from eflow._hidden.objects import FileOutput
from eflow.utils.pandas_utils import descr_table,value_counts_table
from eflow.utils.image_utils import create_plt_png, df_to_image


class DataAnalysis(FileOutput):
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


    def perform_analysis(self,
                         df,
                         df_features):
       try:
           # Iterate through DataFrame columns and graph based on data types
           for col_feature_name in df.columns:
               print(f"Generating graph for {col_feature_name}...\n")

               feature_values = df[col_feature_name].value_counts().keys()
               if len(feature_values) <= 3 and \
                       not col_feature_name in df_features.get_numerical_features():

                   self.pie_graph(df,
                                  col_feature_name,
                                  init_default_color="#C0C0C0")

               elif col_feature_name in df_features.get_categorical_features():
                   self.count_plot_graph(df, col_feature_name)

               elif col_feature_name in df_features.get_integer_features():
                   if len(feature_values) <= 13:
                       self.count_plot_graph(df,
                                             col_feature_name)
                   else:
                       self.distance_plot_graph(df,
                                                col_feature_name)

               elif col_feature_name in df_features.get_float_features():
                   self.distance_plot_graph(df,
                                            col_feature_name)

               plt.close()
           else:
               print(
                   "Object didn't receive a Pandas Dataframe object or a DataFrameTypes object")
       finally:
           self.__called_from_perform = False

    def distance_plot_graph(self,
                            df,
                            col_feature_name,
                            display_table=True):
        """
        df:
            Pandas DataFrame object.

        col_feature_name:
            Specified feature column name.

        display_table:
            Display table in python notebook.

        Returns/Descr:
            Display a distance plot and save the graph/table in the correct
            directory.
        """

        if not self.__called_from_perform:
            if np.sum()

        # Set general graph info
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        plt.title("Distance Plot: " + col_feature_name)

        # Create seaborn graph
        sns.distplot(df[col_feature_name].dropna())

        # Generate image in proper directory structure
        create_plt_png(self.get_output_folder(),
                       "Feature data_analysis/Graphics",
                       "Distance_Plot_" + col_feature_name)
        if self.__notebook_mode:
            plt.show()
        plt.close()

        # Numerical column's multi-metric evaluation
        self.create_descr_table(df,
                                col_feature_name,
                                format_float_pos=3,
                                display_table=display_table)

    def count_plot_graph(self,
                         df,
                         col_feature_name,
                         flip_axis=False,
                         palette="PuBu",
                         display_table=True):
        """
        df:
            Pandas DataFrame object.

        col_feature_name:
            Specified feature column name.

        display_table:
            Display table in python notebook.

        flip_axis:
            Flip the x and y axis for visual representation.

        palette:
            Seaborn color palette, specifies the colors the graph will use.

        Returns/Descr:
            Display a count plot and save the graph/table in the correct
            directory.
        """

        # Set general graph info
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        plt.title("Category Count Plot: " + col_feature_name)

        # Find and rank values based on counts for color variation of the graph
        groupedvalues = df.groupby(col_feature_name).sum().reset_index()

        pal = sns.color_palette(palette, len(groupedvalues))

        rank_list = []
        for target_value in df[col_feature_name].dropna().unique():
            rank_list.append(sum(
                df[col_feature_name] == target_value))

        rank_list = np.argsort(-np.array(rank_list)).argsort()

        # Flip the graph for visual flare
        if flip_axis:
            ax = sns.countplot(y=col_feature_name, data=df,
                               palette=np.array(pal[::-1])[rank_list])
        else:
            ax = sns.countplot(x=col_feature_name, data=df,
                               palette=np.array(pal[::-1])[rank_list])

        # Labels for numerical count of each bar
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1}'.format(height),
                    ha="center")

        # Save graph in proper directory structure
        create_plt_png(self.get_output_folder(),
                       "Feature data_analysis/Graphics",
                       "Count_Plot_" + col_feature_name)
        if self.__notebook_mode:
            plt.show()
        plt.close()

        # Save table in proper directory structure
        self.create_value_counts_table(df,
                                       col_feature_name,
                                       format_float_pos=3,
                                       display_table=display_table)

    def pie_graph(self,
                  df,
                  col_feature_name,
                  colors=None,
                  display_table=True,
                  init_default_color=None):
        """
       df:
           Pandas DataFrame object.

       col_feature_name:
           Specified feature column name.

       colors:
            Accepts an array of hex colors with the correct count of values
            within the feature. If not init; then specified colors will be
            assigned based on if the feature is Boolean or if the column name
            is found in 'defined_column_colors'; else just init with
            random colors.

       display_table:
           Display table in python notebook.

       init_default_color:
           A default color to assign unknown values when other values are
           already assigned. Left to 'None' will init with random colors.

       Returns/Descr:
           Display a pie graph and save the graph/table in the correct
           directory.
        """
        # Find value counts
        value_counts = df[col_feature_name].dropna().value_counts()
        value_list = value_counts.index.tolist()
        value_count_list = value_counts.values.tolist()

        # Init with proper color hex value based on conditionals. (Read Above)
        if colors is None:
            colors = self.__check_specfied_column_colors(df,
                                                         col_feature_name,
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

        # Set general graph info
        fig = plt.gcf()
        fig.set_size_inches(12, 8)
        plt.title("Pie Chart: " + col_feature_name)
        plt.legend(fancybox=True)
        plt.axis('equal')
        plt.tight_layout()
        plt.figure(figsize=(20, 20))

        # Save graph in proper directory structure
        create_plt_png(self.get_output_folder(),
                       "Feature data_analysis/Graphics",
                       "Pie_Chart_" + col_feature_name)
        if self.__notebook_mode:
            plt.show()
        plt.close()

        # Save table in proper directory structure
        self.create_value_counts_table(df,
                                       col_feature_name,
                                       format_float_pos=3,
                                       display_table=display_table)

    def create_value_counts_table(self,
                                  df,
                                  feature_name,
                                  format_float_pos=None,
                                  display_table=True):
        """
        df:
            Pandas DataFrame object.

        col_feature_name:
            Specified feature column name.

        format_float_pos:
            Any features that are floats will be rounded to the
            specified decimal place.

        display_table:
            Display table in python notebook.

        Returns/Descr:
            Returns back a value counts DataFrame table and saves the png in
            the right directory.
        """
        col_vc_df = value_counts_table(df,
                                       feature_name)

        # Convert DataFrame table to image
        df_to_image(col_vc_df,
                    self.get_output_folder(),
                    "Feature data_analysis/Tables/Value Counts",
                    feature_name + "_Value_Counts",
                    show_index=True,
                    format_float_pos=format_float_pos)

        plt.close()

    def create_descr_table(self,
                           df,
                           feature_name,
                           format_float_pos=None,
                           display_table=True):
        """
        df:
            Pandas DataFrame object.

        feature_name:
            Specified feature column name.

        format_float_pos:
            Any features that are floats will be rounded to the
            specified decimal place.

        display_table:
            Display table in python notebook.

        Returns/Descr:
            Returns back a numerical summary DataFrame table and saves the png
            in the right directory.
        """

        # Numerical summary of the column stored in a DataFrame Object.
        col_desc_df = descr_table(df,feature_name)

        if display_table and self.__notebook_mode:
            display(col_desc_df)
            print("\n"*3)

        # Convert DataFrame table to image
        df_to_image(col_desc_df,
                    self.get_output_folder(),
                    "Feature data_analysis/Tables/Descriptions",
                    col_feature_name + "_Descr",
                    show_index=True,
                    format_float_pos=format_float_pos)
        plt.close()

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