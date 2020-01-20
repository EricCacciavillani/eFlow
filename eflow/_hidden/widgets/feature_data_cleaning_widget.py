from eflow._hidden.constants import SYS_CONSTANTS
from eflow.utils.sys_utils import write_object_text_to_file
from eflow.utils.misc_utils import string_condtional

import pandas as pd

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout
import json
import uuid
import os.path
import copy


class DataCleaningWidget():

    def __init__(self,
                 require_input=None,
                 data_cleaning_options=None):
        """
        df: pd.Dataframe
            Pandas dataframe object

        project_name: string
            Appending directory structure/name to the absolute path of the
            output directory.

        overwrite_full_path: string
            Define the entire output path of the cleaner.
        """

        # Throw errors here
        if data_cleaning_options is None:
            return

        if require_input is None:
            return

        # Written conditionals for functions requiring input fields
        self.__require_input = require_input
        self.__data_cleaning_options = data_cleaning_options

        # Define needed widgets
        self.__feature_cleaning_options_w = None
        self.__features_w = None
        self.__options_w = None
        self.__input_w = None
        self.__selected_options_w = None
        self.__full_widgets_ui = None

        # ---
        self.__df_features = None
        self.__json_file_path = None
        self.__feature_input_holder = dict()
        self.__feature_zscore_holder = dict()
        self.__selected_options = dict()

    def get_user_inputs(self):
        return self.__selected_options, \
               {feature: value for feature, value in
                self.__feature_input_holder.items() if value}, \
               {feature: value for feature, value in
                self.__feature_zscore_holder.items() if value}

    def run_widget(self,
                   nan_feature_names,
                   df_features):
        """
        Desc:
            Spin up widget display for in jupyter UI.

        Args:
            df: pd.Dataframe
                A pandas dataframe object

            df_features: DataFrameTypes
                DataFrameTypes object; organizes feature types into groups.

        Returns/Descr:
            Returns a UI widget to create a JSON file for cleaning.
        """
        if not nan_feature_names:
            print("Nan features must not be an empty list!")
            return None

        self.__df_features = df_features

        for feature_name in nan_feature_names:
            if feature_name not in self.__selected_options:
                self.__selected_options[feature_name] = "Ignore feature"
        # self.__selected_options = {feature_name: "Ignore feature"
        #                            for feature_name in nan_feature_names}

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

        self.__selected_options_w = widgets.Select(
            options=list(self.__format_selected_options()),
            layout=Layout(width='80%',
                          height='375px')
        )

        self.__init_update_updateable_widgets()


        self.__input_w = widgets.Text(
            value='',
            placeholder='Replace Value',
            description='Input:',
            disabled=False,
            visible=False,
            layout=Layout(left='49.06%',
                          bottom='250px')
        )

        # ---
        self.__zscore_w = widgets.Text(
            value='',
            placeholder='Z-Score Value',
            description='Z Score:',
            disabled=False,
            visible=False,
            layout=Layout(left='49%',
                          bottom="330px", )
        )

        # Link functions with non-updateable widgets
        self.__features_w.observe(self.__select_feature,
                                  'value')
        self.__features_w.observe(self.__hide_init_zscore_w,
                                  'value')

        self.__input_visible = False
        self.__zscore_visible = False

        self.__zscore_w.observe(self.__validate_save_zscore)
        self.__input_w.observe(self.__validate_save_input)
        self.__options_w.observe(self.__set_input_value,
                                 "value")
        self.__features_w.observe(self.__set_zscore_value,
                                  "value")

        self.__validate_save_zscore(None)
        self.__validate_save_input(None)

        # Setup and display full UI
        self.__full_widgets_ui = widgets.interactive(
            self.__save_option,
            Features=self.__features_w,
            Options=self.__options_w,
            Text_Input=self.__input_w,
            Z_Score_Input=self.__zscore_w,
            Selected=self.__selected_options_w
        )

        display(self.__full_widgets_ui)

    def __init_update_updateable_widgets(self):
        init = self.__features_w.value
        self.__options_w = self.__feature_cleaning_options_w[init]

        if self.__input_w:
            self.__hide_init_input_w(None)

    def __validate_save_input(self,
                              _):
        """
        Returns/Descr:
            Ensures the input field is within specified parameters defined
            by the 'require_input' dictionary.
        """
        #
        # if len(self.__input_w.value) == 0:
        #     return
        #
        # if self.__features_w.value in self.__feature_input_holder and self.__feature_input_holder[self.__features_w.value]:
        #     self.__feature_input_holder.pop(self.__features_w.value,
        #                                     None)

        if self.__options_w.value in self.__require_input:

            feature_type = self.__get_dtype_key(
                    self.__df_features,
                    self.__features_w.value)
            if len(self.__input_w.value) > 0:
                self.__input_w.value = ''.join(
                    [i for i in self.__input_w.value if i.isdigit() or i == '.'
                     or i == "-"])

                if self.__require_input[self.__options_w.value] is not None \
                        and not string_condtional(self.__input_w.value,
                                                  self.__require_input[
                                                      self.__options_w.value]) and \
                        (self.__input_w.value.strip() != "-" and self.__input_w.value.strip() != "."):
                    self.__input_w.value = self.__input_w.value[:-1]

            self.__feature_input_holder[self.__features_w.value] = \
                self.__input_w.value
        else:
            self.__feature_input_holder[self.__features_w.value] = ""
            self.__input_w.value = ""

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
            # if self.__zscore_w.value[-1] == ".":
            #     self.__zscore_w.value += "0"

            self.__feature_zscore_holder[self.__features_w.value] = \
                self.__zscore_w.value

    def __hide_init_input_w(self,
                           _):

        if self.__options_w.value in self.__require_input:
            self.__input_w.layout.visibility = 'visible'
            self.__input_visible = True
        else:
            self.__input_w.layout.visibility = 'hidden'
            self.__input_visible = False

    def __hide_init_zscore_w(self,
                             _):
        if self.__get_dtype_key(self.__df_features,
                                self.__features_w.value) == "Number":
            self.__zscore_w.layout.visibility = "visible"
            self.__zscore_visible = True
        else:
            self.__zscore_w.layout.visibility = "hidden"
            self.__zscore_visible = False

    def __save_option(self,
                      **func_kwargs):
        if not self.__selected_options:
            return None

        self.__selected_options[func_kwargs["Features"]] = self.__options_w.value
        self.__selected_options_w.options = self.__format_selected_options()

        # print(func_kwargs["Features"])
        # print("\n\n\t     Feature option Review\n\t   " + "*" * 25)
        # for feature, option in self.__selected_options.items():
        #
        #     if option[0:3] == "---":
        #         option = "Ignore Feature"
        #
        #     print("\n\t   Feature: {0}\n\t   Option:  {1}\n".format(
        #         feature,
        #         option)
        #           + "           " + "----" * 7)
        self.__hide_init_input_w(None)

    def __set_input_value(self,
                          _):

        if self.__input_visible:
            if self.__features_w.value in self.__feature_input_holder:
                self.__input_w.value = self.__feature_input_holder[self.__features_w.value]
            else:
                self.__input_w.value = ""
        else:
            self.__input_w.value = ""
            if self.__features_w.value in self.__feature_input_holder:
                self.__feature_input_holder[self.__features_w.value] = ""


    def __set_zscore_value(self,
                          _):

        if self.__zscore_visible:
            if self.__features_w.value in self.__feature_zscore_holder:
                self.__zscore_w.value = self.__feature_zscore_holder[self.__features_w.value]
            else:
                self.__zscore_w.value = ""
        else:
            self.__zscore_w.value = ""
            if self.__features_w.value in self.__feature_zscore_holder:
                self.__feature_zscore_holder[self.__features_w.value] = ""


    def __select_feature(self,
                         feature):
        """
        Returns/Descr:
            When a feature selection is chosen all the widgets are
            re-initialized.
        """

        write_object_text_to_file(self.__feature_input_holder,
                                  os.getcwd(),
        "fuckk")

        self.__init_update_updateable_widgets()

        new_i = widgets.interactive(self.__save_option,
                                    Features=self.__features_w,
                                    Options=self.__options_w,
                                    Text_Input=self.__input_w,
                                    Z_Score_Input=self.__zscore_w,
                                    Selected=self.__selected_options_w)

        self.__full_widgets_ui.children = new_i.children

        self.__set_input_value("really?")

    # --- General functionality

    def __format_selected_options(self):

        formated_list = []
        for feature, option in self.__selected_options.items():

            if option[0:3] in "---":
                option = "Ignore Feature"

            formated_string = "Feature:{:<20s} Option:{:s}".format(feature,
                                                                   option)

            tmp = str()
            for i, char in enumerate(formated_string):

                if i == len(formated_string)-1:
                    tmp += char
                    break

                if char == " " and formated_string[i - 1] != ":" and formated_string[i - 1] == " " and \
                        formated_string[i + 1] == " ":
                    tmp += "-"
                else:
                    tmp += char
            formated_string = tmp

            formated_list.append(formated_string)

        return formated_list

    def __get_dtype_key(self,
                        df_features,
                        col_feature_name):
        """
        Args:
            df_features:
                DataFrameTypes object; organizes feature types into groups.

            col_feature_name:
                Pandas column name.

        Returns/Descr:
            Returns back the data type of the feature that is created
        """

        if col_feature_name in df_features.continuous_numerical_features():
            return "Number"
        elif col_feature_name in df_features.categorical_features() or col_feature_name in df_features.string_features():
            return "Category"
        elif col_feature_name in df_features.bool_features():
            return "Bool"
        else:
            return "Unknown"