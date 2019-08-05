import pandas as pd
import missingno as msno

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout
import json
from scipy import stats
import uuid
import os.path

from eFlow.Utils.Sys_Utils import *
from eFlow.Utils.Constants import *


class DataCleaner:

    def __init__(self,
                 df=None,
                 project_name="Default_Data_Cleaner",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 missing_data_visuals=True):
        """

        df:
            Pandas dataframe object

        project_name:
            Appending directory structure/name to the absolute path of the
            output directory.

        overwrite_full_path:
            Define the entire output path of the cleaner.

        notebook_mode:
            Notebook mode is to display images/tables in jupyter notebooks.

        missing_data_visuals:
            Provide visual for viewing any missing data.
        """

        self.__requires_nan_removal = df.isnull().values.any()
        self.__notebook_mode = notebook_mode

        # Setup project structure
        if not overwrite_full_path:
            parent_structure = "/" + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME \
                               + "/" + project_name + "/"
            self.__PROJECT = enum(PATH_TO_OUTPUT_FOLDER=
                os.getcwd()  + parent_structure)
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

        ### Setting up widget options

        # Dummy line to show in the menu for cleaner viewing
        # self.__data_cleaning_options["TYPE"][
        #     "---------------------" + (" " * space_counters.pop())] = \
        #     self.__ignore_feature

        # Set up numerical cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options = dict()
        self.__data_cleaning_options["Number"] = dict()

        self.__data_cleaning_options["Number"]["Ignore feature"] = \
            self.__ignore_feature
        self.__data_cleaning_options["Number"]["Drop feature"] = \
            self.__drop_feature
        self.__data_cleaning_options["Number"]["Remove all nans"] = \
            self.__remove_nans

        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"]["Ignore feature"] = \
            self.__ignore_feature
        self.__data_cleaning_options["Number"][
            "Fill nan with min value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with x% value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with median value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with max value of distribution"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with x% value of distribution after removing outliers"] = \
            self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill nan with average value of distribution"] = self.__fill_nan_by_distribution
        self.__data_cleaning_options["Number"]["Peform interpolation"] = \
            self.__peform_interpolation
        self.__data_cleaning_options["Number"][
            "Fill null with specfic value"] = self.__fill_nan_with_specfic_value
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

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

        # Set up category cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Category"] = dict()
        self.__data_cleaning_options["Category"]["Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Category"]["Fill nan with x% value of distribution"] = self.__ignore_feature
        self.__data_cleaning_options["Category"]["Drop feature"] = \
            self.__drop_feature

        # Set up boolean cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Bool"] = dict()
        self.__data_cleaning_options["Bool"][
            "Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Bool"]["Drop feature"] = \
            self.__drop_feature

        # Error case on data types
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Unknown"] = dict()
        self.__data_cleaning_options["Unknown"][
            "ERROR UNKNOWN FEATURE TYPE FOUND"] = self.__ignore_feature
        self.__data_cleaning_options["Unknown"][
            "Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Unknown"]["Drop feature"] = \
            self.__ignore_feature

        # Written conditionals for functions requiring input fields
        self.__require_input = {"Fill null with specfic value":None,
                                "Fill nan with x% value of distribution":
                                    'x >= 0 and x <=100',
                                "Fill nan with x% value of distribution "
                                "after removing outliers":'x >= 0 and x <=100'}

        # Define needed widgets
        self.__feature_cleaning_options_w = None
        self.__features_w = None
        self.__options_w = None
        self.__text_w = None
        self.__full_widgets_ui = None
        self.__submit_button = None

        # ---
        self.__selected_options = None
        self.__tmp_df_features = None
        self.__last_saved_json_file_path = None
        self.__feature_input_holder = dict()
        self.__zscore_feature_holder = dict()


    def data_cleaning_widget(self,
                             df,
                             df_features):
        """
        df:
            A pandas dataframe object

        df_features:
            DataFrameTypes object; organizes feature types into groups.

        Returns/Descr:
            Returns a UI widget to create a JSON file for cleaning.
        """
        self.__tmp_df_features = df_features
        self.__feature_input_holder = dict()

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
            layout=Layout(width='50%',
                          height='175px')
        )

        init = self.__features_w.value
        self.__options_w = self.__feature_cleaning_options_w[init]

        # Init widgets/interactables
        self.__text_w = widgets.Text(
            value='',
            placeholder='Replace Value',
            description='Input:',
            disabled=False,
            visible=False,
            layout=Layout(left='510px',
                          bottom='250px')
        )

        # ---
        self.__file_name_w = widgets.Text(
            value='Default Data Cleaning',
            placeholder='Replace Value',
            description='File Name:',
            disabled=False,
            visible=False,
            layout=Layout(left='590px')
        )

        # ---
        self.__zscore_w =  widgets.Text(
            value='',
            placeholder='Z-Score Value',
            description='Z Score:',
            disabled=False,
            visible=False,
            layout=Layout(left='510px',
                          bottom="330px",)
        )

        # ---
        self.__submit_button = widgets.Button(
            description='Create JSON File from options',
            color="#ff1122",
            layout=Layout(left='100px',
                          bottom="5px",
                          width='40%', ))

        # Link functions with interactables
        self.__features_w.observe(self.__data_cleaning_widget_select_feature,
                                  'value')
        self.__zscore_w.observe(self.__validate_save_zscore)
        self.__file_name_w.observe(self.__validate_file_name)
        self.__text_w.observe(self.__validate_save_text_w)
        self.__submit_button.on_click(self.__create_data_cleaning_json_file)
        self.__options_w.observe(self.__hide_init_text_w,
                                 "new")

        # Setup and display full UI
        self.__full_widgets_ui = widgets.interactive(
            self.__data_cleaning_widget_save_option,
            Features=self.__features_w,
            Options=self.__options_w,
            Text_Input=self.__text_w,
            Z_Score_Input=self.__zscore_w,
        )
        display(self.__file_name_w)
        display(self.__full_widgets_ui)
        display(self.__submit_button)

        return self.__submit_button

    def data_cleaning_with_json_file(self,
                                     df,
                                     json_file_path):
        """
        df:
            Pandas dataframe object.
        json_file_path:
            Path to JSON object file for cleaning

        Returns/Descr:
            Peforms cleaning of the dataframe with the proper json object.
        """
        with open(json_file_path) as json_file:
            data = json.load(json_file)
            for feature, json_obj in data.items():
                print(feature)
                print(json_obj["Option"])
                print(self.__data_cleaning_options[json_obj["Type"]][
                    json_obj["Option"]])
                self.__data_cleaning_options[json_obj["Type"]][
                    json_obj["Option"]](df,
                                        json_obj)
                print()

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

    ### Widget controllers ###
    def __validate_file_name(self,
                             _):
        """
        Returns/Descr:
            Ensures the filename can actually be created based on the widgets
            input and if no filename is given then a random filename that
            doesn't exist in the given directory will be init in the widget.
        """

        # Remove characters that aren't alphanumeric or the given characters
        self.__file_name_w.value = "".join(x for x in str(
            self.__file_name_w.value) if x.isalnum() or x == "_" or x == "("
                                           or x == ")" or x == " " or x == "-")

        # Generate a random unique filename in the directory structure.
        if self.__file_name_w.value == "":

            json_path = self.__PROJECT.PATH_TO_OUTPUT_FOLDER[0:
                                                             self.__PROJECT.PATH_TO_OUTPUT_FOLDER.find(
                                                                 SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME)] + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME
            check_create_dir_structure(json_path,
                                       "JSON File Dump/Data Cleaning")

            # Keep generating a filename until a unique one is found
            tmp_file_name = uuid.uuid4().hex.upper()[0:16]
            while(os.path.isfile(json_path + '/JSON File Dump/Data Cleaning/' +
                              tmp_file_name + ".json")):
                tmp_file_name = uuid.uuid4().hex.upper()[0:16]

            # ---
            self.__file_name_w.value = tmp_file_name

    def __validate_save_text_w(self,
                               _):
        """
        Returns/Descr:
            Ensures the input field is within specified parameters defined
            by the 'require_input' dictionary.
        """

        if self.__options_w.value in self.__require_input:
            self.__feature_input_holder[self.__features_w.value] = \
                self.__text_w.value

            feature_type = self.__get_dtype_key(
                    self.__tmp_df_features,
                    self.__features_w.value)
            if feature_type == "Number" and len(self.__text_w.value):

                self.__text_w.value = ''.join(
                    [i for i in self.__text_w.value if i.isdigit() or i == '.'])

                if self.__require_input[self.__options_w.value] is not None \
                        and not string_condtional(float(self.__text_w.value),
                                                  self.__require_input[
                                                      self.__options_w.value]):
                    self.__text_w.value = self.__text_w.value[:-1]

        else:
            if self.__features_w.value in self.__feature_input_holder:
                self.__feature_input_holder.pop(self.__features_w.value,
                                                None)

    def __validate_save_zscore(self,
                               _):
        """
        Returns/Descr:
            Validates the z-score widget and saves the value of the z-score
            with selected feature.
        """

        self.__zscore_w.value = "".join(x for x in str(
            self.__zscore_w.value) if x.isdigit() or x == ".")

        if self.__zscore_w.value == ".":
            self.__zscore_w.value = "0.0"

        if self.__zscore_w.value != "":
            if self.__zscore_w.value[-1] == ".":
                self.__zscore_w.value += "0"

            self.__zscore_feature_holder[self.__features_w.value] = \
                self.__zscore_w.value
        else:
            self.__zscore_feature_holder[self.__features_w.value] = None

    def __hide_init_text_w(self,
                           _):

        if self.__options_w.value in self.__require_input:
            self.__text_w.value = ""
            self.__text_w.layout.visibility = 'visible'
        else:
            self.__text_w.layout.visibility = 'hidden'

        if self.__features_w.value in self.__feature_input_holder:
            self.__text_w.value = self.__feature_input_holder[
                self.__features_w.value ]

    def __data_cleaning_widget_save_option(self,
                                           **func_kwargs):
        if not self.__selected_options:
            return None

        self.__selected_options[func_kwargs["Features"]] = self.__options_w.value

        print(func_kwargs["Features"])
        print("\n\n\t     Feature option Review\n\t   " + "*" * 25)
        for feature, option in self.__selected_options.items():

            if option[0:3] == "---":
                option = "Ignore Feature"

            print("\n\t   Feature: {0}\n\t   Option:  {1}\n".format(
                feature,
                option)
                  + "           " + "----" * 7)

        self.__hide_init_text_w(None)

    def __create_data_cleaning_json_file(self,
                                         _):
        """
        Returns/Descr:
            Generates the json file with feature to cleaning options
            and any extra data.
        """

        json_dict = dict()
        for feature, option in self.__selected_options.items():

            json_dict[feature] = dict()
            json_dict[feature]["Type"] = self.__get_dtype_key(
                self.__tmp_df_features,
                feature)
            json_dict[feature]["Option"] = option
            json_dict[feature]["Extra"] = dict()

            if option == "Fill null with specfic value" and feature in self.__feature_input_holder.keys():
                json_dict[feature]["Extra"]["Replace value"] = self.__feature_input_holder[feature]
            elif option == "Fill nan with x% value of distribution" and feature in self.__feature_input_holder.keys():
                json_dict[feature]["Extra"]["Percentage of distribution"] = self.__feature_input_holder[feature]

        json_path = self.__PROJECT.PATH_TO_OUTPUT_FOLDER[0:
                                                         self.__PROJECT.PATH_TO_OUTPUT_FOLDER.find(
                                                             SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME)] + SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME

        json_dict[feature]["Extra"] = json_path

        check_create_dir_structure(json_path,
                                   "JSON File Dump/Data Cleaning")

        abs_file_path = json_path + "/JSON File Dump/Data Cleaning/" + (self.__file_name_w.value) + ".json"

        with open(abs_file_path,
                  'w',
                  encoding='utf-8') as outfile:
            json.dump(json_dict,
                      outfile,
                      ensure_ascii=False,
                      indent=2)

        self.__last_saved_json_file_path = abs_file_path

    def __data_cleaning_widget_select_feature(self,
                                              feature):
        """
        Returns/Descr:
            When a feature selection is chosen all the widgets are
            re-initialized.
        """

        self.__text_w = widgets.Text(
            value='',
            placeholder='Replace Value',
            description='Input:',
            disabled=False,
            visible=False,
            layout=Layout(left='510px',
                          bottom='250px')
        )
        self.__zscore_w = widgets.Text(
            value='',
            placeholder='Z-Score Value',
            description='Z Score:',
            disabled=False,
            visible=False,
            layout=Layout(left='510px',
                          bottom="330px", )
        )
        self.__text_w.observe(self.__validate_save_text_w)
        self.__zscore_w.observe(self.__validate_save_zscore)

        if self.__get_dtype_key(self.__tmp_df_features,
                                feature["new"]) == "Number":
            self.__zscore_w.layout.visibility = "visible"
        else:
            self.__zscore_w.layout.visibility = "hidden"

        if feature["new"] in self.__zscore_feature_holder and \
                self.__zscore_feature_holder[feature["new"]]:
            self.__zscore_w.value = str(
                self.__zscore_feature_holder[feature["new"]])

        write_object_to_file("testing.txt",
                             [feature["new"],
                              self.__selected_options])

        self.__options_w = self.__feature_cleaning_options_w[feature["new"]]
        self.__options_w.observe(self.__hide_init_text_w,
                                 "new")

        new_i = widgets.interactive(self.__data_cleaning_widget_save_option,
                                    Features=self.__features_w,
                                    Options=self.__options_w,
                                    Text_Input=self.__text_w,
                                    Z_Score_Input=self.__zscore_w)

        self.__full_widgets_ui.children = new_i.children


    def __get_dtype_key(self,
                        df_features,
                        col_feature_name):
        """
        df_features:
            DataFrameTypes object.
        col_feature_name:
            Pandas column name.

        Returns/Descr:
            Returns back the data type of the feature that is created
        """

        if col_feature_name in df_features.get_numerical_features():
            return "Number"
        elif col_feature_name in df_features.get_categorical_features():
            return "Category"
        elif col_feature_name in df_features.get_bool_features():
            return "Bool"
        else:
            return "Unknown"

    ### Cleaning options ###
    def __ignore_feature(self,
                         df,
                         json_obj):
        """
        Do nothing to this feature for nan removal
        """
        print("Ignoring Feature: ", json_obj["Feature"])

    def __drop_feature(self,
                       df,
                       json_obj):
        print("Droping Feature: ", json_obj["Feature"])
        df.drop(columns=json_obj["Feature"],
                inplace=True)
        df.reset_index(drop=True,
                       inplace=True)

    def __remove_nans(self,
                      df,
                      json_obj):
        print("Removing Nans: ", json_obj["Feature"])
        df[json_obj["Feature"]].dropna(inplace=True)
        df.reset_index(drop=True,
                       inplace=True)


    def __zcore_remove_outliers(self,
                                df,
                                feature):

        z_score_return = stats.zscore(((df[feature].dropna())))
        return df[feature].dropna()[
            (z_score_return >= -2) & (z_score_return <= 2)]


    def __fill_nan_by_distribution(self,
                                   df,
                                   json_obj):

        print("Fill nan by distribution for feature {0} by ".format(
            json_obj["Feature"]),
              end='')

        if "removing outliers" in json_obj["Option"]:
            series_obj = self.__zcore_remove_outliers(df,
                                                      json_obj["Feature"])
        else:
            series_obj = df[json_obj["Feature"]].dropna()

        if "min" in json_obj["Option"]:
            fill_na_val = np.percentile(series_obj, 0)
            print("the minimum")
        elif "median" in json_obj["Option"]:
            fill_na_val = np.percentile(series_obj, 50)
            print("the median")
        elif "max" in json_obj["Option"]:
            fill_na_val = np.percentile(series_obj, 100)
            print("the maximum")
        else:
            fill_na_val = np.percentile(series_obj,
                                        float(
                                            json_obj["Extra"]
                                            ["Percentage of distribution"]))
            print("{0}%".format(float(json_obj["Extra"]
                                      ["Percentage of distribution"])))

        df[json_obj["Feature"]].fillna(fill_na_val,
                                       inplace=True)

    def __fill_nan_by_average(self,
                              df,
                              json_obj):

        print("Fill nan by average ",end='')

        if "removing outliers" in json_obj["Option"]:
            print("after removing outliers by feature: {0}".format(
                json_obj["Feature"]))
            series_obj = self.__zcore_remove_outliers(df,
                                                      json_obj["Feature"])
        else:
            print("by feature: {0}".format(json_obj["Feature"]))
            series_obj = df[json_obj["Feature"]].dropna()

        df[json_obj["Feature"]].fillna(
            series_obj.mean(),
            inplace=True)

    def __peform_interpolation(self,
                               df,
                               json_obj):
        print("peform interpolation")
        pass

    def __fill_nan_with_specfic_value(self,
                                      df,
                                      json_obj):
        print("Replace nan with {0} on feature: {1}".format(json_obj["Extra"]["Replace value"],
                                         json_obj["Feature"]))

        if json_obj["Type"] == "Number":

            if "." in json_obj["Extra"]["Replace value"]:
                fill_nan_val = float(json_obj["Extra"]["Replace value"])
            else:
                fill_nan_val = int(json_obj["Extra"]["Replace value"])
        else:
            fill_nan_val = json_obj["Extra"]["Replace value"]
        df[json_obj["Feature"]].fillna(fill_nan_val,
                                       inplace=True)

    def __fill_nan_by_count_distrubtion(self,
                                        df,
                                        json_obj):
        print("nan by count")
        pass

    ### Getters ###
    def get_last_saved_json_file_path(self):
        return copy.deepcopy(self.__last_saved_json_file_path)