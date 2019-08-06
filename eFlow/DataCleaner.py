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
from eFlow.Widgets.DataCleaningWidget import *

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

        # Throw error here
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

            print(self.__PROJECT.PATH_TO_OUTPUT_FOLDER)

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
        self.__data_cleaning_options["Category"]["Fill null with specfic value"] = self.__fill_nan_with_specfic_value
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

        # ---
        self.__notebook_mode = notebook_mode
        self.__ui_widget = None


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
        self.__ui_widget = None

        # Re-init for multiple
        if self.__notebook_mode:
            self.__ui_widget = DataCleaningWidget(
                require_input=self.__require_input,
                data_cleaning_options=self.__data_cleaning_options,
                overwrite_full_path=self.__PROJECT.PATH_TO_OUTPUT_FOLDER)
            self.__ui_widget.run_widget(df,
                                 df_features)
        else:
            print("Can't initialize widget in a non-notebook space")


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
        return copy.deepcopy(self.__ui_widget.get_last_saved_json_file_path())