from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from IPython.display import display, HTML
import os
import copy
import random


class DataAnalysis:

    def __init__(self,
                 df,
                 df_features,
                 project_name="Default",
                 overwrite_figure_path=None):

        self.__defined_column_colors = list()

        self.__defined_column_colors.append([["gender", "sex"],
                                             ["Male", "M", "#7EAED3"],
                                             ["Female", "F", "#FFB6C1"]])

        def enum(**enums):
            return type('Enum', (), enums)

        if not overwrite_figure_path:
            overwrite_figure_path = "/Figures/" + project_name + "/"

        self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=''.join(
            os.getcwd().partition('/Libraries')[0:1]) + overwrite_figure_path)

        display(df.dtypes)
        print("\n\n")
        display(self.missing_values_table(df))
        print("\n")

        if df.isnull().values.any():

            # ---
            msno.matrix(df)
            plt.show()
            plt.close()

            # ---
            msno.bar(df, color="#072F5F")
            plt.show()
            plt.close()

            # ---
            msno.heatmap(df)
            plt.show()
            plt.close()

            # ---
            msno.dendrogram(df)
            plt.show()
            plt.close()

        print("*" * 80 + "\n" * 2)

        for col_feature_name in df.columns:
            if col_feature_name in df_features.get_bool_features() or len(
                    df[col_feature_name].value_counts().values) == 2:
                self.pie_graph(df, col_feature_name)
            elif col_feature_name in df_features.get_categorical_features():
                self.count_plot_graph(df, col_feature_name)
            elif col_feature_name in df_features.get_integer_features():
                if len(df[col_feature_name].dropna().unique()) <= 12:
                    self.count_plot_graph(df,col_feature_name)
                else:
                    self.distance_plot(df,col_feature_name)
#                 print(df[col_feature_name].describe())
#                 print(df[col_feature_name].var())
#                 print("\n\n\n\n\n")
            elif col_feature_name in df_features.get_float_features():
                self.distance_plot(df,col_feature_name)
#                 print(df[col_feature_name].describe())
#                 print(df[col_feature_name].var())
#                 print("\n\n\n\n\n")

    def distance_plot(self,
                      df,
                      col_feature_name):
        sns.set(style="whitegrid")

        plt.figure(figsize=(12, 8))
        plt.title("Distance Plot: " + col_feature_name)
        sns.distplot(df[col_feature_name].dropna())

        self.__create_plt_png("Data_Analysis_Quick_Look",
                              "Distance Plot: " + col_feature_name)
        plt.show()
        plt.close()
    def __check_specfied_column_colors(self,
                                       df,
                                       col_feature_name):
        
        for column_info in copy.deepcopy(self.__defined_column_colors):

            specified_column_names = column_info.pop(0)
            specified_column_names = {x.upper()
                                      for x in specified_column_names}

            if col_feature_name.upper() in specified_column_names:

                column_colors = list()
                df_specfied_column_values = [
                    x.upper() for x in df[col_feature_name].dropna().unique()]

                for column_value_info in column_info:
                    column_value_color = column_value_info.pop(-1)

                    selected_color = False
                    for test_column_value in {x for x in column_value_info}:
                        if test_column_value.upper() in df_specfied_column_values:
                            column_colors.append(column_value_color)
                            selected_color = True

                    # Add a random hex color value if none are chose
                    if not selected_color:
                        column_colors.append(
                            "#%06x" %
                            random.randint(
                                0, 0xFFFFFF))
                return column_colors

        return None

    def count_plot_graph(self,
                         df,
                         col_feature_name,
                         flip_axis=False,
                         pallete="PuBu"):

        sns.set(style="whitegrid")
        plt.figure(figsize=(12, 8))

        groupedvalues = df.groupby(col_feature_name).sum().reset_index()

        pal = sns.color_palette(pallete, len(groupedvalues))

        rank_list = []
        for target_value in df[col_feature_name].unique():

            if pd.isna(target_value) or target_value == "Nan".upper(
            ) == target_value.upper() or target_value == " " or target_value == "":
                continue
            rank_list.append(sum(df[col_feature_name].dropna() == target_value))

        rank_list = np.argsort(-np.array(rank_list)).argsort()

        if flip_axis:
            ax = sns.countplot(y=col_feature_name, data=df,
                               palette=np.array(pal[::-1])[rank_list])
        else:
            ax = sns.countplot(x=col_feature_name, data=df,
                               palette=np.array(pal[::-1])[rank_list])

        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2.,
                    height + 3,
                    '{:1}'.format(height),
                    ha="center")

        plt.title("Category Count Plot: " + col_feature_name)
        self.__create_plt_png("Data Analysis: Quick Look",
                              "Count Plot: " + col_feature_name)
        plt.show()
        plt.close()

    def pie_graph(self,
                  df,
                  col_feature_name,
                  colors=None):

        value_list = df[col_feature_name].unique()

        target_count_list = []
        for target_value in value_list:
            target_count_list.append(sum(df[col_feature_name].dropna() == target_value))
        
        if colors is None:
            
            colors = []
            if df[col_feature_name].dtypes.name == 'bool':
                for val in value_list:
                    if val:
                        colors.append("#57ff57")
                    else:
                        colors.append("#ff8585")
            else:
                colors = self.__check_specfied_column_colors(df, col_feature_name)

        explode_array = [0] * len(value_list)
        explode_array[np.array(target_count_list).argmax()] = .03

        plt.pie(
            tuple(target_count_list),
            labels=tuple(value_list),
            shadow=False,
            colors=colors,
            explode=tuple(explode_array),  # Space between slices
            startangle=90,    # Rotate conter-clockwise by 90 degrees
            autopct='%1.1f%%',  # Display fraction as percentage
        )
        fig = plt.gcf()
        fig.set_size_inches(8, 7)
        plt.title("Pie Chart: " + col_feature_name)
        plt.legend(fancybox=True)
        plt.axis('equal')
        plt.tight_layout()
        plt.figure(figsize=(20, 20))
        self.__create_plt_png("Data Analysis: Quick Look",
                              "Pie Chart: " + col_feature_name)
        plt.show()
        plt.close()

    def missing_values_table(self, df):
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
        print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
              "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

    def __check_create_figure_dir(self,
                                  sub_dir):
        """
            Checks/Creates required directory structures inside
            the parent directory figures.
        """

        directory_pth = self.__PROJECT.PATH_TO_OUTPUT_FOLDER

        for dir in sub_dir.split("/"):
            directory_pth += "/" + dir
            if not os.path.exists(directory_pth):
                os.makedirs(directory_pth)

        return directory_pth

    def __create_plt_png(self,
                         sub_dir,
                         filename):
        """
            Saves the plt based image in the correct directory.
        """

        # Ensure directory structure is init correctly
        abs_path = self.__check_create_figure_dir(sub_dir)

        # Ensure file ext is on the file.
        if filename[-4:] != ".png":
            filename += ".png"

        fig = plt.figure(1)
        fig.savefig(abs_path + "/" + filename, bbox_inches='tight')
