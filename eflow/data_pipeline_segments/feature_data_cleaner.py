from eflow._hidden.widgets.feature_data_cleaning_widget import *
from eflow._hidden.parent_objects.data_pipeline_segment import *
from eflow.foundation import DataFrameTypes
from eflow.utils.pandas_utils import check_if_feature_exists
from eflow.utils.misc_utils import get_parameters

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

import pandas as pd
import numpy as np
from scipy import stats
from collections import deque


class FeatureDataCleaner(DataPipelineSegment):
    """
    Designed for a multipurpose data cleaner.
    """

    def __init__(self,
                 segment_id=None,
                 create_file=True):
        """
        Args:
            segment_id: str
                Reference id to past segments of this object.

        Note/Caveats:
            When creating any public function that will be part of the pipeline's
            structure it is important to follow this given template. Also,
            try not to use _add_to_que. Can ruin the entire purpose of this
            project.
        """
        DataPipelineSegment.__init__(self,
                                     object_type=self.__class__.__name__,
                                     segment_id=segment_id,
                                     create_file=create_file)

        self.__test_cleaning_methods = False

        # self.__requires_nan_removal = df.isnull().values.any()
        #
        # # Throw error here
        # if df is None:
        #     return

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
            self.ignore_feature
        self.__data_cleaning_options["Number"]["Drop feature"] = \
            self.drop_feature
        self.__data_cleaning_options["Number"]["Remove all nans"] = \
            self.remove_nans

        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill nan with min value of distribution"] = \
            self.fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with x% value of distribution"] = \
            self.fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with median value of distribution"] = \
            self.fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "Fill nan with max value of distribution"] = \
            self.fill_nan_by_distribution
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill nan with average value of distribution"] = self.fill_nan_by_average
        self.__data_cleaning_options["Number"][
            "Fill nan with mode of distribution"] = self.fill_nan_by_mode
        self.__data_cleaning_options["Number"][
            "Fill null with specfic value"] = self.fill_nan_with_specfic_value
        self.__data_cleaning_options["Number"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.ignore_feature

        self.__data_cleaning_options["Number"][
            "Fill with least common count of distribution"] = \
            self.fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Number"][
            "Fill with most common count of distribution"] = \
            self.fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Number"][
            "Fill with x% count distribution"] = \
            self.fill_nan_by_occurance_percentaile

        # Set up category cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Category"] = dict()
        self.__data_cleaning_options["Category"]["Ignore feature"] = self.ignore_feature
        self.__data_cleaning_options["Category"]["Drop feature"] = \
            self.drop_feature
        self.__data_cleaning_options["Category"]["Remove all nans"] = \
            self.remove_nans

        self.__data_cleaning_options["Category"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.ignore_feature
        self.__data_cleaning_options["Category"][
            "Fill null with specfic value"] = self.fill_nan_with_specfic_value
        self.__data_cleaning_options["Category"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.ignore_feature

        self.__data_cleaning_options["Category"][
            "Fill nan with mode of distribution"] = self.fill_nan_by_mode
        self.__data_cleaning_options["Category"][
            "Fill with least common count of distribution"] = \
            self.fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Category"][
            "Fill with most common count of distribution"] = \
            self.fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Category"][
            "Fill with x% count distribution"] = \
            self.fill_nan_by_occurance_percentaile


        # Set up boolean cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Bool"] = dict()
        self.__data_cleaning_options["Bool"][
            "Ignore feature"] = self.ignore_feature
        self.__data_cleaning_options["Bool"]["Drop feature"] = \
            self.drop_feature
        self.__data_cleaning_options["Bool"]["Remove all nans"] = \
            self.remove_nans

        self.__data_cleaning_options["Bool"][
            "---------------------" + (" " * space_counters.pop())] = \
            self.ignore_feature

        self.__data_cleaning_options["Bool"]["Make nan Assertions"] = self.make_nan_assertions

        self.__data_cleaning_options["Bool"][
            "Fill null with specfic bool value"] = self.fill_nan_with_specfic_value

        self.__data_cleaning_options["Bool"][
            "Fill with least common count of distribution"] = \
            self.fill_nan_by_occurance_percentaile
        self.__data_cleaning_options["Bool"][
            "Fill with most common count of distribution"] = \
            self.fill_nan_by_occurance_percentaile

        # Set up boolean cleaning options
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Date"] = dict()
        self.__data_cleaning_options["Date"][
            "Ignore feature"] = self.ignore_feature
        self.__data_cleaning_options["Date"]["Drop feature"] = \
            self.drop_feature

        # Error case on data types
        space_counters = {i for i in range(1, 50)}
        self.__data_cleaning_options["Unknown"] = dict()
        self.__data_cleaning_options["Unknown"][
            "ERROR UNKNOWN FEATURE TYPE FOUND"] = self.ignore_feature
        self.__data_cleaning_options["Unknown"][
            "Ignore feature"] = self.ignore_feature
        self.__data_cleaning_options["Unknown"]["Drop feature"] = \
            self.ignore_feature
        self.__data_cleaning_options["Unknown"]["Remove all nans"] = \
            self.remove_nans

        # Written conditionals for functions requiring input fields
        self.__require_input = {"Fill null with specfic value": None,
                                "Fill null with specfic bool value": 'x == 1 or x == 0',
                                "Fill nan with x% value of distribution":
                                    'x >= 0 and x <=100',
                                "Fill with random existing values": 'x > 0',
                                "Fill with x% count distribution":
                                    'x >= 0 and x <=100'}

        # ---
        # self.__notebook_mode = notebook_mode
        self.__ui_widget = DataCleaningWidget(
            require_input=self.__require_input,
            data_cleaning_options=self.__data_cleaning_options)


    def get_user_inputs(self):
        return self.__ui_widget.get_user_inputs()

    def run_widget(self,
                   df,
                   df_features,
                   nan_feature_names=[]):
        """
        df:
            A pandas dataframe object

        df_features:
            DataFrameTypes object; organizes feature types into groups.

        Returns/Descr:
            Returns a UI widget to create a JSON file for cleaning.
        """

        # Throw Error here
        if df is None:
            return

        if not nan_feature_names:
            nan_feature_names = df.columns[df.isna().any()].tolist()

        self.__ui_widget.run_widget(nan_feature_names,
                                    df_features)

    def perform_saved_widget_input(self,
                                   df,
                                   df_features,
                                   suppress_runtime_errors=True,
                                   reset_segment_file=False):

        try:
            for bool_value in [True,False]:
                self.__test_cleaning_methods = bool_value

                selected_options, \
                feature_input_holder, \
                feature_zscore_holder = self.__ui_widget.get_user_inputs()

                for feature_name, function_option in selected_options.items():

                    for dtype in ["Number","Bool","Category", "Date"]:

                        if function_option in self.__data_cleaning_options[dtype]:

                            saved_function = self.__data_cleaning_options[dtype][function_option]
                            exec_str = f"saved_function(df,df_features,feature_name,"

                            if feature_name in feature_input_holder and \
                                    feature_input_holder[feature_name]:
                                    try:
                                        int(feature_input_holder[feature_name])
                                        exec_str += feature_input_holder[
                                                        feature_name] + ","
                                    except:
                                        exec_str += "\"" + feature_input_holder[feature_name] + "\","

                            if function_option == "Fill nan with min value of distribution" or function_option == "Fill with least common count of distribution":
                                exec_str += "0,"
                            elif function_option == "Fill nan with median value of distribution":
                                exec_str += "50,"
                            elif function_option == "Fill nan with max value of distribution" or function_option == "Fill with most common count of distribution":
                                exec_str += "100,"

                            if feature_name in feature_zscore_holder and \
                                    feature_zscore_holder[feature_name]:
                                exec_str += feature_zscore_holder[feature_name] + ","

                            exec_str += ")"

                            if self.__test_cleaning_methods:
                                defined_function_string = exec_str.replace("saved_function",
                                                                           saved_function.__name__).replace("feature_name",
                                                                                                            f"\"{feature_name}\"")
                                print("***" * 10)
                                print("Testing function")
                                print(defined_function_string)
                            # print(exec_str)
                            try:
                                exec(exec_str)

                                if self.__test_cleaning_methods:
                                    print("PASSED TEST!")
                                    print("***" * 10)
                                print()
                            except Exception as e:

                                if self.__test_cleaning_methods:
                                    print("FAILED TEST!")
                                    print("***" * 10)
                                self.__test_cleaning_methods = False
                                if reset_segment_file:
                                    print("Exception hit when trying to perform all "
                                          "cleaning functions. "
                                          "Resetting json object for feature data cleaner segment!")
                                    self.reset_segment_file()
                                raise e
                            break
        finally:
            self.__test_cleaning_methods = False

    # --- Cleaning options
    def make_nan_assertions(self,
                            df,
                            df_features,
                            feature_name,
                            _add_to_que=True):
        """
        Desc:
            Make nan assertions for boolean features.

        Args:
            df: pd.Dataframe
                Pandas Dataframe

            df_features: DataFrameType from eflow
                Organizes feature types into groups.

            feature_name: string
                Name of the feature in the datatframe

            _add_to_que: bool
                Pushes the function to pipeline segment parent if set to 'True'.
        """

        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")


        if feature_name not in df_features.bool_features():
            raise UnsatisfiedRequirments(f"{feature_name} must be a bool feature.")

        unique_series = df[feature_name].dropna().unique().tolist()

        if len(unique_series) == 1 and (
                unique_series[0] == 1 or unique_series[0] == 0):
            replace_value = int(unique_series[0] == 1)

            self.fill_nan_with_specfic_value(df,
                                             df_features,
                                             feature_name=feature_name,
                                             replace_value=replace_value,
                                             _add_to_que=_add_to_que)

        else:
            raise UnsatisfiedRequirments(f"Boolean assertions can't be made with this given feature {feature_name}.")

    def ignore_feature(self,
                       df,
                       df_features,
                       feature_name,
                       _add_to_que=True):
        """
        Desc:
            Ignore the given feature.

        Args:
            df: pd.Dataframe
                Pandas Dataframe

            df_features: DataFrameType from eflow
                Organizes feature types into groups.

            feature_name: string
                Name of the feature in the datatframe

            _add_to_que: bool
                Pushes the function to pipeline segment parent if set to 'True'.
        """
        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        if not self.__test_cleaning_methods:
            print("Ignore feature: ", feature_name)

            if _add_to_que:
                params_dict = locals()
                parameters = get_parameters(self.drop_feature)

                self._DataPipelineSegment__add_function_to_que("ignore_feature",
                                                               parameters,
                                                               params_dict)

            # Remove any unwanted arguments in params_dict
            # if _add_to_que:
            #     params_dict = locals()
            #     parameters = get_parameters(self.ignore_feature)
            #
            #     self._DataPipelineSegmentfunction_name("ignore_feature",
            #                                            parameters,
            #                                            params_dict)


    def drop_feature(self,
                     df,
                     df_features,
                     feature_name,
                     _add_to_que=True):
        """
        Desc:
            Drop a feature in the dataframe.

        Args:
            df: pd.Dataframe
                Pandas Dataframe

            df_features: DataFrameType from eflow
                Organizes feature types into groups.

            feature_name: string
                Name of the feature in the datatframe

            _add_to_que: bool
                Pushes the function to pipeline segment parent if set to 'True'.
        """
        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        if not self.__test_cleaning_methods:

            print("Droping Feature: ", feature_name)

            df.drop(columns=feature_name,
                    inplace=True)
            df.reset_index(drop=True,
                           inplace=True)
            df_features.remove_feature(feature_name)

            if _add_to_que:
                params_dict = locals()
                parameters = get_parameters(self.drop_feature)

                self._DataPipelineSegment__add_function_to_que("drop_feature",
                                                               parameters,
                                                               params_dict)

    def remove_nans(self,
                    df,
                    df_features,
                    feature_name,
                    _add_to_que=True):
        """
        Desc:
            Remove rows of data based on the given feature.

        Args:
            df: pd.Dataframe
                Pandas Dataframe

            df_features: DataFrameType from eflow
                Organizes feature types into groups.

            feature_name: string
                Name of the feature in the datatframe

            _add_to_que: bool
                Pushes the function to pipeline segment parent if set to 'True'.
        """
        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        if not self.__test_cleaning_methods:

            print(f"Remove data from rows where the feature {feature_name} is equal to nan")

            df[feature_name].dropna(inplace=True)
            df.reset_index(drop=True,
                           inplace=True)
            df_features.remove_feature(feature_name)

            if _add_to_que:

                # Remove any unwanted arguments in params_dict
                params_dict = locals()
                parameters = get_parameters(self.remove_nans)

                self._DataPipelineSegment__add_function_to_que("remove_nans",
                                                               parameters,
                                                               params_dict)


    def fill_nan_by_distribution(self,
                                 df,
                                 df_features,
                                 feature_name,
                                 percentile,
                                 z_score=None,
                                 _add_to_que=True):
        """
        Desc:
            Fill nan by the distribution of data.

        Args:
            df: pd.Dataframe
                Pandas Dataframe

            df_features: DataFrameType from eflow
                Organizes feature types into groups.

            feature_name: string
                Name of the feature in the datatframe

            percentile: float or int

            z_score:

            _add_to_que: bool
                Pushes the function to pipeline segment parent if set to 'True'.
        """
        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        if feature_name in df_features.continuous_numerical_features():
            series_obj = df[feature_name].sort_values()
        else:
            series_obj = df.sort_values([feature_name],
                                        ascending=True).groupby(feature_name).head(float("inf"))[feature_name]

        if z_score:
            if isinstance(z_score, float) or isinstance(z_score, int):
                series_obj = self.__zcore_remove_outliers(series_obj.to_frame(),
                                                          feature_name,
                                                          z_score)
            else:
                raise ValueError("Z-Score must be at numerical value.")
        else:
            series_obj = df[feature_name].dropna()

        replace_value = np.percentile(series_obj, percentile)

        # Remove any unwanted arguments in params_dict

        if not self.__test_cleaning_methods:
            print(f"Fill nan on distribution; {percentile}% of {feature_name}")
            self.fill_nan_with_specfic_value(df,
                                             df_features,
                                             feature_name=feature_name,
                                             replace_value=replace_value,
                                             _add_to_que=_add_to_que)

    def fill_nan_by_average(self,
                            df,
                            df_features,
                            feature_name,
                            z_score=None,
                            _add_to_que=True):

        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        params_dict = locals()

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self", "df", "df_features", "_add_to_que",
                        "params_dict"]:
                del params_dict[arg]

        if feature_name not in df_features.continuous_numerical_features():
            raise UnsatisfiedRequirments(f"{feature_name} must be a saved as float or integer in df_features")

        if z_score:
            if isinstance(z_score,float) or isinstance(z_score,int):
                series_obj = self.__zcore_remove_outliers(df,
                                                          feature_name,
                                                          z_score)
            else:
                raise ValueError("Z-Score must be at numerical value.")
        else:
            series_obj = df[feature_name].dropna()

        replace_value = series_obj.mean()

        if not self.__test_cleaning_methods:
            print("Fill nan based on the average of the distribution.")
            self.fill_nan_with_specfic_value(df,
                                             df_features,
                                             feature_name=feature_name,
                                             replace_value=replace_value,
                                             _add_to_que=_add_to_que)

    def fill_nan_by_mode(self,
                         df,
                         df_features,
                         feature_name,
                         z_score=None,
                         _add_to_que=True):

        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        if z_score:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature_name,
                                                      z_score)
        else:
            series_obj = df[feature_name].dropna()

        mode_series = series_obj.mode()
        if not len(mode_series):
            pass
        else:
            replace_value = mode_series[0]

        if not self.__test_cleaning_methods:
            print("Fill nan by mode")
            self.fill_nan_with_specfic_value(df,
                                             df_features,
                                             feature_name=feature_name,
                                             replace_value=replace_value,
                                             _add_to_que=_add_to_que)

    def fill_nan_with_specfic_value(self,
                                    df,
                                    df_features,
                                    feature_name,
                                    replace_value,
                                    _add_to_que=True):

        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        try:
            replace_value = replace_value.item()
        except AttributeError:
            pass

        if not self.__test_cleaning_methods:
            df[feature_name].fillna(replace_value,
                                    inplace=True)

            if _add_to_que:
                params_dict = locals()
                parameters = get_parameters(self.fill_nan_with_specfic_value)

                print("Replace nan with {0} on feature: {1}".format(replace_value,
                                                                    feature_name))

                self._DataPipelineSegment__add_function_to_que("fill_nan_with_specfic_value",
                                                               parameters,
                                                               params_dict)


    def fill_nan_by_occurance_percentaile(self,
                                          df,
                                          df_features,
                                          feature_name,
                                          percentaile,
                                          z_score=None,
                                          _add_to_que=True):
        check_if_feature_exists(df,
                                feature_name)

        if feature_name not in df_features.all_features():
            raise KeyError(
                f"The feature \'{feature_name}\' was not found in the dataframe!"
                + " Please select a valid feature from the df_features.")

        # Remove any unwanted arguments in params_dict
        if _add_to_que:
            params_dict = locals()
            for arg in ["self", "df", "df_features", "_add_to_que",
                        "params_dict"]:
                try:
                    del params_dict[arg]
                except KeyError:
                    pass

        if z_score:
            series_obj = self.__zcore_remove_outliers(df,
                                                      feature_name,
                                                      z_score)
        else:
            series_obj = df[feature_name].dropna()

        array = np.asarray(series_obj.value_counts() / df.dropna().shape[0])
        idx = (np.abs(array - (percentaile / 100))).argmin()
        replace_value = series_obj.value_counts().keys()[idx]

        if not self.__test_cleaning_methods:
            print("Fill nan by occurance percentaile.")
            self.fill_nan_with_specfic_value(df,
                                             df_features,
                                             feature_name=feature_name,
                                             replace_value=replace_value,
                                             _add_to_que=_add_to_que)

    def __zcore_remove_outliers(self,
                                df,
                                feature_name,
                                zscore_val,
                                _add_to_que=True):

        z_score_return = 2

        print("After the zscore applied of {0} to -{0}".format(zscore_val))

        return df[feature_name].dropna()[
            (z_score_return >= (zscore_val * -1)) & (
                    z_score_return <= zscore_val)]


