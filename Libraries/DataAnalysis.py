from matplotlib import pyplot as plt
from pandas.plotting import table
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from IPython.display import display, HTML
import os
import copy
import random
import imgkit
import six

from Libraries.Sys_Utils import *


class DataAnalysis:

    def __init__(self,
                 df=None,
                 df_features=None,
                 project_name="Default_Project_Name",
                 overwrite_full_path=None):
        """
        df:
            Pandas DataFrame object.

        df_features:
            DataframeTypeHolder object. Contains the DataFrame's types
            to create better code/workflow.

        project_name:
            Creates a parent or "project" folder in which all sub-directories
            will be inner nested.

        overwrite_full_path:
            Overwrites the path to the parent folder.

        Returns/Descr:
            Designed to increase workflow and overall attempt to automate the
            routine of graphing/generating tables.

        Personal Note:
            I will continue to expand on different ways of exploring data and
            automating that process.
        """

        # Pre-defined colors for column's with set names.
        # Multiple names/values are allowed
        self.__defined_column_colors = list()
        self.__defined_column_colors.append([["gender", "sex"],
                                             ["Male", "M", "#7EAED3"],
                                             ["Female", "F", "#FFB6C1"]])
        self.__defined_column_colors.append([[" "],
                                             ["Y", "yes", "#55a868"],
                                             ["N", "No", "#ff8585"]])
        self.__defined_column_colors.append([[" "],
                                             [True, "#55a868"],
                                             [False, "#ff8585"]])


        if not overwrite_full_path:
            parent_structure = "/Production Data/" + project_name + "/"
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=''.join(
                os.getcwd().partition('/Libraries')[0:1]) + parent_structure)
        else:
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=overwrite_full_path)

        if df is not None and df_features is not None:
            # ---
            display(df.dtypes)
            print("\n\n")
            display(self.__missing_values_table(df))
            print("\n")

            if df.isnull().values.any():

                # ---
                msno.matrix(df)
                create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                               "Missing_Data/Graphics",
                               "Missing_Data_Matrix_Graph")
                plt.show()
                plt.close()

                # ---
                msno.bar(df,
                         color="#072F5F")

                create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                               "Missing_Data/Graphics",
                               "Missing_Data_Bar_Graph")
                plt.show()
                plt.close()

                # ---
                msno.heatmap(df)
                create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                               "Missing_Data/Graphics",
                               "Missing_Data_Heatmap")
                plt.show()
                plt.close()

                # ---
                msno.dendrogram(df)
                create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                               "Missing_Data/Graphics",
                               "Missing_Data_Dendrogram_Graph")
                plt.show()
                plt.close()

            print("*" * 80 + "\n" * 2)

            # Iterate through DataFrame columns and graph based on data types
            for col_feature_name in df.columns:

                if len(df[col_feature_name].value_counts().values) <= 3 and \
                        not col_feature_name in df_features.get_numerical_features():

                    self.pie_graph(df,
                                   col_feature_name,
                                   init_default_color="#C0C0C0")

                elif col_feature_name in df_features.get_categorical_features():
                    self.count_plot_graph(df, col_feature_name)

                elif col_feature_name in df_features.get_integer_features():
                    if len(df[col_feature_name].dropna().unique()) <= 19:
                        self.count_plot_graph(df,
                                              col_feature_name)
                    else:
                        self.distance_plot_graph(df,
                                                 col_feature_name)

                elif col_feature_name in df_features.get_float_features():
                    self.distance_plot_graph(df,
                                             col_feature_name)

                print("-" * 80 + "\n" * 2)
        else:
            print("Object didn't receive a Pandas Dataframe object or a DataFrameTypes object")

    def __missing_values_table(self,
                               df):
        """

        df:
            Pandas DataFrame object

        Returns/Descr:
            Returns/Saves a Pandas DataFrame object giving the percentage of the
            null data for the original DataFrame columns.
        """
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
                                                                  "There are " + str(
            mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")

        # ---
        df_to_image(mis_val_table_ren_columns,
                    self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                    "Missing_Data/Tables",
                    "Missing_Data_Table",
                    show_index=True,
                    format_float_pos=2)

        return mis_val_table_ren_columns

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

        # Set general graph info
        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))
        plt.title("Distance Plot: " + col_feature_name)

        # Create seaborn graph
        sns.distplot(df[col_feature_name].dropna())

        # Generate image in proper directory structure
        create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                       "Feature_Analysis/Graphics",
                       "Distance_Plot_" + col_feature_name)
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
        create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                       "Feature_Analysis/Graphics",
                       "Count_Plot_" + col_feature_name)
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
        value_counts = df[col_feature_name].value_counts()
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
        create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                       "Feature_Analysis/Graphics",
                       "Pie_Chart_" + col_feature_name)
        plt.show()
        plt.close()

        # Save table in proper directory structure
        self.create_value_counts_table(df,
                                       col_feature_name,
                                       format_float_pos=3,
                                       display_table=display_table)

    def create_value_counts_table(self,
                                  df,
                                  col_feature_name,
                                  format_float_pos=None,
                                  display_table=True
                                  ):
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

        # Value counts DataFrame
        col_vc_df = df[col_feature_name].value_counts().rename_axis(
            'Unique Values').reset_index(name='Counts')

        if display_table:
            display(col_vc_df)
            print("\n"*3)

        # Convert DataFrame table to image
        df_to_image(col_vc_df,
                    self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                    "Feature_Analysis/Tables",
                    col_feature_name + " Value Counts",
                    show_index=True,
                    format_float_pos=format_float_pos)

    def create_descr_table(self,
                           df,
                           col_feature_name,
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
            Returns back a numerical summary DataFrame table and saves the png
            in the right directory.
        """

        # Numerical summary of the column stored in a DataFrame Object.
        col_desc_df = df[col_feature_name].describe().to_frame()
        col_desc_df["var"] = df[col_feature_name].var()

        if display_table:
            display(col_desc_df)
            print("\n"*3)

        # Convert DataFrame table to image
        df_to_image(col_desc_df,
                    self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                    "Feature_Analysis/Tables",
                    col_feature_name + " Descr",
                    show_index=True,
                    format_float_pos=format_float_pos)

    def __check_specfied_column_colors(self,
                                       df,
                                       col_feature_name,
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
            df[col_feature_name].value_counts().index.tolist()]

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
            if col_feature_name.upper() in specified_column_names or \
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

