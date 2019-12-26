from eflow._hidden.widgets.feature_data_cleaning_widget import *
from eflow._hidden.parent_objects.data_pipeline_segment import *
from eflow.foundation import DataFrameTypes

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

import pandas as pd
import numpy as np


class DataCleaner(DataPipelineSegment):
    """
    Designed for a multipurpose data cleaner.
    """

    def __init__(self,
                 df=None,
                 project_name="Data Cleaning",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 missing_data_visuals=True,
                 make_nan_assertions=True):
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

        FileOutput.__init__(self,
                            project_name,
                            overwrite_full_path)

        # Throw error here
        if df is None:
            return

        # --- Setting up widget options

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
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill nan with average value of distribution"] = self.__fill_nan_by_average
        self.__data_cleaning_options["Number"][
            "Fill nan with mode of distribution"] = self.__fill_nan_by_mode
        self.__data_cleaning_options["Number"][
            "Fill null with specfic value"] = self.__fill_nan_with_specfic_value
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.__ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill with least common count of distribution"] = \
            self.__fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Number"][
            "Fill with most common count of distribution"] = \
            self.__fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Number"][
            "Fill with x% count distribution"] = \
            self.__fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Number"]["Fill with random existing values"] = \
            self.__fill_nan_with_existing_values

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

        # Set up boolean cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Date"] = dict()
        self.__data_cleaning_options["Date"][
            "Ignore feature"] = self.__ignore_feature
        self.__data_cleaning_options["Date"]["Drop feature"] = \
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
        self.__data_cleaning_options["Unknown"]["Remove all nans"] = \
            self.__remove_nans

        # Written conditionals for functions requiring input fields
        self.__require_input = {"Fill null with specfic value":None,
                                "Fill nan with x% value of distribution":
                                    'x >= 0 and x <=100',
                                "Fill with random existing values": 'x > 0',
                                "Fill with x% count distribution":
                                    'x >= 0 and x <=100'}

        # ---
        self.__notebook_mode = notebook_mode
        self.__ui_widget = None

        if make_nan_assertions:
            df_features = DataFrameTypes(df,
                                         display_init=False)
            self.__make_nan_assertions(df,
                                       df_features)

    def init_json_file_name(self,
                            filename):

        if not isinstance(filename,str):
            print("THROW ERROR Filename must be a string")
        filename = filename.split(".", 1)[0]
        filename += ".json"

        self.__filename = filename

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
                overwrite_full_path=FileOutput.get_output_folder(self))
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
                self.__data_cleaning_options[json_obj["Type"]][
                    json_obj["Option"]](df,
                                        feature,
                                        json_obj)
                print("**"*30 + "\n")

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
                    FileOutput.get_output_folder(self),
                    "Missing_Data/Tables",
                    "Missing_Data_Table",
                    show_index=True,
                    format_float_pos=2)

        return mis_val_table_ren_columns

    def __make_nan_assertions(self,
                              df,
                              df_features):

        for bool_feature in df_features.bool_features():
            if len(df[bool_feature].dropna().value_counts().values) != 2:
                print("testing")

    # --- Cleaning options
    def __zcore_remove_outliers(self,
                                df,
                                feature,
                                zscore_val):

        z_score_return = stats.zscore(((df[feature].dropna())))
        return df[feature].dropna()[
            (z_score_return >= (zscore_val * -1)) & (
                    z_score_return <= zscore_val)]

    def __ignore_feature(self,
                         df,
                         feature,
                         json_obj):
        """
            Do nothing to this feature for nan removal
        """
        print("Ignoring Feature: ", feature)

    def __drop_feature(self,
                       df,
                       feature,
                       json_obj):
        print("Droping Feature: ", feature)
        df.drop(columns=feature,
                inplace=True)
        df.reset_index(drop=True,
                       inplace=True)

    def __remove_nans(self,
                      df,
                      json_obj):
        print("Removing Nans: ",feature)
        df[feature].dropna(inplace=True)
        df.reset_index(drop=True,
                       inplace=True)

    def __fill_nan_by_distribution(self,
                                   df,
                                   feature,
                                   json_obj):

        print("Fill nan by distribution")

        if "Zscore" in json_obj["Extra"]:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature,
                                                      json_obj["Extra"][
                                                          "Zscore"])
        else:
            series_obj = df[feature].dropna()

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
                                            ["Input"]))
            print("{0}%".format(float(json_obj["Extra"]
                                      ["Input"])))

        if "Zscore" in json_obj["Extra"]:
            print("After the zscore applied of {0} to -{0}".format(
                json_obj["Extra"]["Zscore"]))

        print("Replace nan with {0} on feature: {1}".format(
            fill_na_val,
            feature))

        df[feature].fillna(fill_na_val,
                           inplace=True)

    def __fill_nan_by_average(self,
                              df,
                              feature,
                              json_obj):

        print("Fill nan by average")

        if "Zscore" in json_obj["Extra"]:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature,
                                                      json_obj["Extra"][
                                                          "Zscore"])
        else:
            series_obj = df[feature].dropna()

        replace_value = series_obj.mean()

        print("Replace nan with {0} on feature: {1}".format(
            replace_value,
            feature))

        if "Zscore" in json_obj["Extra"]:
            print("After the zscore applied of {0} to -{0}".format(
                json_obj["Extra"]["Zscore"]))

        df[feature].fillna(
            replace_value,
            inplace=True)

    def __fill_nan_by_mode(self,
                           df,
                           feature,
                           json_obj):

        print("Fill nan by mode")
        if "Zscore" in json_obj["Extra"]:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature,
                                                      json_obj["Extra"][
                                                          "Zscore"])
        else:
            series_obj = df[feature].dropna()

        replace_value = series_obj.mode()[0]

        print("Replace nan with {0} on feature: {1}".format(
            replace_value,
            feature))

        if "Zscore" in json_obj["Extra"]:
            print("After the zscore applied of {0} to -{0}".formt(
                json_obj["Extra"]["Zscore"]))

        df[feature].fillna(
            replace_value,
            inplace=True)

    def __fill_nan_with_specfic_value(self,
                                      df,
                                      feature,
                                      json_obj):

        print("Replace nan with {0} on feature: {1}".format(
            json_obj["Extra"]["Replace value"],
            feature))

        fill_nan_val = json_obj["Extra"]["Input"]
        df[feature].fillna(fill_nan_val,
                                       inplace=True)

    def __fill_nan_by_occurance_percentaile(self,
                                            df,
                                            feature,
                                            json_obj):

        print("Fill nan by occurance percentaile")

        if "Zscore" in json_obj["Extra"]:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature,
                                                      json_obj["Extra"][
                                                          "Zscore"])
        else:
            series_obj = df[feature].dropna()

        if "most" in json_obj["Option"]:
            target_value = 100
        elif "least" in json_obj["Option"]:
            target_value = 0
        else:
            target_value = json_obj["Extra"]["Input"]

        array = np.asarray(series_obj.value_counts() / df.shape[0])
        idx = (np.abs(array - target_value)).argmin()
        replace_value = series_obj.value_counts().keys()[idx]

        print("Replace nan with {0} on feature: {1}".format(
            replace_value,
            feature))

        if "Zscore" in json_obj["Extra"]:
            print("After the zscore applied of {0} to -{0}".formt(
                json_obj["Extra"]["Zscore"]))

        df[feature].fillna(replace_value,
                           inplace=True)

    def __fill_nan_with_existing_values(self,
                                        df,
                                        feature,
                                        json_obj):
        print("Fill nan with random existing values on feature {0}".format(feature))

        if "Zscore" in json_obj["Extra"]:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature,
                                                      json_obj["Extra"][
                                                          "Zscore"])
        else:
            series_obj = df[feature].dropna()

        if "Zscore" in json_obj["Extra"]:
            print("After the zscore applied of {0} to -{0}".formt(
                json_obj["Extra"]["Zscore"]))
        df[feature].fillna(
            pd.Series(np.random.choice(list(series_obj.value_counts().keys()),
                                       size=len(df.index))))

    # Getters
    def json_file_path(self):

        # self.__json_filename =
        self.__json_file_path = self.__ui_widget.get_last_saved_json_file_path()
        return copy.deepcopy()