import numpy as np
import pandas as pd
import missingno as msno
import asyncio

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Button, Layout

from Libraries.Sys_Utils import *

class DataCleaner:

    def __init__(self,
                 df=None,
                 df_features=None,
                 project_name="Default_Project_Name_Data_Cleaner",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 missing_data_graphing=True):

        self.__requires_nan_removal = df.isnull().values.any()
        self.__notebook_mode = notebook_mode

        if not overwrite_full_path:
            parent_structure = "/Production Data/" + project_name + "/"
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=''.join(
                os.getcwd().partition('/Libraries')[0:1]) + parent_structure)
        else:
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=overwrite_full_path)

        if self.__requires_nan_removal and missing_data_graphing:

            # ---
            msno.matrix(df)
            create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                           "Missing_Data/Graphics",
                           "Missing_Data_Matrix_Graph")

            if self.__notebook_mode:
                plt.show()
            plt.close()

            # ---
            msno.bar(df,
                     color="#072F5F")

            create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                           "Missing_Data/Graphics",
                           "Missing_Data_Bar_Graph")
            if self.__notebook_mode:
                plt.show()
            plt.close()

            # ---
            msno.heatmap(df)
            create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                           "Missing_Data/Graphics",
                           "Missing_Data_Heatmap")
            if self.__notebook_mode:
                plt.show()
            plt.close()

            # ---
            msno.dendrogram(df)
            create_plt_png(self.__PROJECT.PATH_TO_OUTPUT_FOLDER,
                           "Missing_Data/Graphics",
                           "Missing_Data_Dendrogram_Graph")
            if self.__notebook_mode:
                plt.show()
            plt.close()

        '''
        Setting up widget options
        '''

        # Dummy line to show in the menu for cleaner viewing
        # self.__data_cleaning_options[""][
        #     "---------------------" + (" " * space_counters.pop())] = \
        #     self.__ignore_feature

        # Set up numerical options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options = dict()
        self.__data_cleaning_options["Number"] = dict()

        self.__data_cleaning_options["Number"]["Ignore feature"] = \
            self.__ignore_feature
        self.__data_cleaning_options["Number"]["Drop feature"] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"]["Ignore feature"] = \
            self.__ignore_feature
        self.__data_cleaning_options["Number"][
            "Fill nan with min value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with 25% value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with median value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with 75% value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with max value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill nan with average value of distribution"] = self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"]["Peform interpolation"] = \
            self.__peform_interpolation
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill null with specfic value"] = self.__fill_nan_with_specfic_value
        self.__data_cleaning_options["Number"][
            "Fill with least common count of distribution"] = \
            self.__fill_nan_by_count_distrubtion

        self.__data_cleaning_options["Number"][
            "Fill with 25% common count of distribution"] = \
            self.__fill_nan_by_count_distrubtion

        self.__data_cleaning_options["Number"][
            "Fill with median common count of distribution"] = \
            self.__fill_nan_by_count_distrubtion

        self.__data_cleaning_options["Number"][
            "Fill with 75% common count of distribution"] = \
            self.__fill_nan_by_count_distrubtion

        self.__data_cleaning_options["Number"][
            "Fill with most common count of distribution"] = \
            self.__fill_nan_by_count_distrubtion

        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Category"] = dict()
        self.__data_cleaning_options["Category"]["Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Category"]["Drop feature"] = \
            self.__ignore_feature

        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Bool"] = dict()
        self.__data_cleaning_options["Bool"][
            "Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Bool"]["Drop feature"] = \
            self.__ignore_feature

        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Unknown"] = dict()
        self.__data_cleaning_options["Unknown"][
            "Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Unknown"]["Drop feature"] = \
            self.__ignore_feature

        self.__selected_options = None
        self.__features_w = None
        self.__options_w = None
        self.__feature_cleaning_options_w = None
        self.__text_w = None
        self.__full_widgets_ui = None
        self.__submit_button = None

    def data_cleaning_widget(self,
                             df,
                             df_features,
                             data_cleaning_csv_path=None):

        if not data_cleaning_csv_path:

            nan_feature_names = df.columns[df.isna().any()].tolist()

            self.__selected_options = {feature_name: "Ignore feature"
                                       for feature_name in nan_feature_names}

            feature_cleaning_options = {col_feature_name:self.__data_cleaning_options[
                self.__get_dtype_key(df_features,
                                     col_feature_name)].keys()
                                        for col_feature_name in nan_feature_names}

            self.__feature_cleaning_options_w = {key: widgets.Select(
                options=feature_cleaning_options[key],
                layout=Layout(width='70%',
                              height='300px'))
                for key in feature_cleaning_options}

            self.__features_w = widgets.Select(
                options=list(feature_cleaning_options.keys()))
            init = self.__features_w.value
            self.__options_w = self.__feature_cleaning_options_w[init]
            self.__features_w.observe(self.__data_cleaning_widget_select_feature,
                                      'value')
            self.__text_w = widgets.Text(
                value='',
                placeholder='Replace Value',
                description='Input:',
                disabled=False,
                visible=False
            )

            self.__submit_button = widgets.Button(description='Run')

            self.__full_widgets_ui = widgets.interactive(
                self.__data_cleaning_widget_save_option,
                Features=self.__features_w,
                Options=self.__options_w,
                Text_Input=self.__text_w)

            display(self.__full_widgets_ui)
            display(self.__submit_button)

        cleaning_dataframe = None
        return self.__submit_button, cleaning_dataframe

    def peform_cleaning(self,
                        df,
                        cleaning_dataframe=None,
                        data_cleaning_csv_path=None):

        if cleaning_dataframe is None and data_cleaning_csv_path is None:
            print("You must past in either a data cleaning csv absolute"
                  "path or a cleaning dataframe.")
            return None

    def __data_cleaning_widget_save_option(self,
                                           **func_kwargs):

        if not self.__selected_options:
            return None

        self.__selected_options[func_kwargs["Features"]] = func_kwargs[
            "Options"]

        if func_kwargs["Options"] == "Fill null with specfic value":
            self.__text_w.layout.visibility = 'visible'
        else:
            self.__text_w.layout.visibility = 'hidden'
            self.__text_w.value = ""

        print("\n\n\t     Feature option Review\n\t   " + "*" * 25)
        for feature, option in self.__selected_options.items():

            if option[0:3] == "---":
                option = "Ignore Feature"

            print("\n\t   Feature: {0}\n\t   Option:  {1}\n".format(
                feature,
                option)
                  + "           " + "----" * 7)

    def __data_cleaning_widget_select_feature(self,
                                              feature):

        print(self.__feature_cleaning_options_w)

        new_i = widgets.interactive(self.__data_cleaning_widget_save_option,
                                    Features=self.__features_w,
                                    Options=self.__feature_cleaning_options_w[
                                        feature['new']])
        self.__full_widgets_ui.children = new_i.children

    def __get_dtype_key(self,
                        df_features,
                        col_feature_name):

        if col_feature_name in df_features.get_numerical_features():
            return "Number"
        elif col_feature_name in df_features.get_categorical_features():
            return "Category"
        elif col_feature_name in df_features.get_bool_features():
            return "Bool"
        else:
            return "Unknown"

    def __ignore_feature(self,
                         args):
        pass

    def __drop_feature(self,
                       args):
        pass

    def __fill_nan_by_distribution(self,
                                   args):
        pass

    def __peform_interpolation(self,
                               args):
        pass

    def __fill_nan_with_specfic_value(self,
                                      args):
        pass

    def __fill_nan_by_count_distrubtion(self,
                                        args):
        pass