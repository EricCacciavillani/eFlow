import pandas as pd
import missingno as msno

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout
import json

from eFlow.Utils.Sys_Utils import *
from eFlow.Utils.Constants import *

class DataCleaner:

    def __init__(self,
                 df=None,
                 project_name="Default_Data_Cleaner",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 missing_data_visuals=True):

        self.__requires_nan_removal = df.isnull().values.any()
        self.__notebook_mode = notebook_mode

        if not overwrite_full_path:
            parent_structure = "/" + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME \
                               + "/" + project_name + "/"
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=''.join(
                os.getcwd().partition('/eFlow')[0:1]) + parent_structure)
        else:
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=overwrite_full_path)

        if df is None:
            return

        if self.__requires_nan_removal and missing_data_visuals:

            display(self.__missing_values_table(df))

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
        self.__tmp_df_features = None

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
                layout=Layout(width='50%',
                              height='300px'))
                for key in feature_cleaning_options}

            self.__features_w = widgets.Select(
                options=list(feature_cleaning_options.keys()),
                layout=Layout(width='50%')
            )
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

            self.__submit_button = widgets.Button(
                description='Create JSON File from options',
                color="#ff1122",
                layout=Layout(left='100px',
                              bottom="5px",
                              width='40%',))

            self.__submit_button.on_click(self.__create_data_cleaning_json_file)

            self.__full_widgets_ui = widgets.interactive(
                self.__data_cleaning_widget_save_option,
                Features=self.__features_w,
                Options=self.__options_w,
                Text_Input=self.__text_w,
             )

            display(self.__full_widgets_ui)
            display(self.__submit_button)

            self.__tmp_df_features = df_features

        return self.__submit_button

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

    def __create_data_cleaning_json_file(self,
                                         b):

        json_dict = dict()
        for feature, option in self.__selected_options.items():

            json_dict[feature] = dict()
            json_dict[feature]["Type"] = self.__get_dtype_key(
                self.__tmp_df_features,
                feature)
            json_dict[feature]["Option"] = option
            json_dict[feature]["Extra"] = dict()

        json_path = self.__PROJECT.PATH_TO_OUTPUT_FOLDER[0:
                                                         self.__PROJECT.PATH_TO_OUTPUT_FOLDER.find(
                                                             SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME)] + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME
        check_create_dir_structure(json_path,
                                   "JSON Files")

        with open(json_path + '/JSON Files/Data_Cleaner.json',
                  'w',
                  encoding='utf-8') as outfile:
            json.dump(json_dict,
                      outfile,
                      ensure_ascii=False,
                      indent=2)

    def __data_cleaning_widget_select_feature(self,
                                              feature):

        new_i = widgets.interactive(self.__data_cleaning_widget_save_option,
                                    Features=self.__features_w,
                                    Options=self.__feature_cleaning_options_w[
                                        feature['new']])
        self.__full_widgets_ui.children = new_i.children

    def data_cleaning_with_json_file(self,
                                     df,
                                     json_file_path):
        with open(json_file_path) as json_file:
            data = json.load(json_file)
            for feature,json_obj in data.items():
                print(feature)
                print(json_obj)
                self.__data_cleaning_options[json_obj["Type"]][json_obj["Option"]]("")
                print("")

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
                         df,
                         json_obj):
        """
        Do nothing to this feature for nan removal
        """

        pass

    def __drop_feature(self,
                       args):
        df.

    def __fill_nan_by_distribution(self,
                                   df,
                                   json_obj):
        pass

    def __peform_interpolation(self,
                               df,
                               json_obj):
        pass

    def __fill_nan_with_specfic_value(self,
                                      df,
                                      json_obj):
        pass

    def __fill_nan_by_count_distrubtion(self,
                                        df,
                                        json_obj):
        pass