from eflow.utils.string_utils import create_hex_decimal_string

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout

import copy

class ColorLabelingWidget():

    def __init__(self):

        self.__features_w = None
        self.__values_w = None
        self.__color_picker_w = None
        self.__full_widgets_ui = None
        self.__feature_values_w = None

        # ---
        self.__feature_value_color_dict = dict()
        self.__init_features_set = set()

    def get_feature_value_color_dict(self):
        return copy.deepcopy(self.__feature_value_color_dict)

    def run_widget(self,
                   feature_value_color_dict):
        """
        df:
            A pandas dataframe object

        df_features:
            DataFrameTypes object; organizes feature types into groups.

        Returns/Descr:
            Returns a UI widget to create a JSON file for cleaning.
        """

        self.__feature_value_color_dict = copy.deepcopy(feature_value_color_dict)

        # Convert all
        for k,v in feature_value_color_dict.items():
            if not v:
                self.__feature_value_color_dict[k] = "#000000"

            if isinstance(v,str):
                del self.__feature_value_color_dict[k]

        feature_values = {feature_name: self.__feature_value_color_dict[feature_name]
                          for feature_name in self.__feature_value_color_dict.keys()}

        self.__features_w = widgets.Select(
            options=list(feature_values.keys()),
            layout=Layout(width='50%',
                          height='175px')
        )

        self.__feature_values_w = {feature: widgets.Select(
             options=feature_values[feature].keys(),
             layout=Layout(width='50%',
                          height='300px'))
             for feature in feature_values}

        self.__color_picker_w = widgets.ColorPicker(
            concise=False,
            description='Pick a color',
            disabled=False
        )

        self.__update_widgets()

        # Link functions with non-updateable widgets
        self.__features_w.observe(self.__select_feature,
                                  'value')

        self.__color_picker_w.observe(self.__select_color,
                                      'value')

        # Setup and display full UI
        self.__full_widgets_ui = widgets.interactive(
            self.__select_value,
            Features=self.__features_w,
            Values=self.__values_w,
            Color_Picker=self.__color_picker_w,
        )

        display(self.__full_widgets_ui)

    def __update_widgets(self):

        init = self.__features_w.value
        self.__values_w = self.__feature_values_w[init]


    def __select_value(self,
                       **func_kwargs):
        """
        Desc:
            Save colors to dictionary and save json file.
        """

        saved_color = self.__feature_value_color_dict[self.__features_w.value][
            self.__values_w.value]

        if not saved_color:
            self.__color_picker_w.value = "#000000"
        else:
            self.__color_picker_w.value = saved_color


    def __select_color(self,
                       _):
        """
        Desc:
            Select the color on color picker widget.
        """

        if self.__color_picker_w.value != "#000000":
            self.__feature_value_color_dict[self.__features_w.value][self.__values_w.value] = self.__color_picker_w.value

    def __select_feature(self,
                         _):
            """
            Desc:
                When a feature selection is chosen all the widgets are
                re-initialized.
            """

            self.__update_widgets()

            new_i = widgets.interactive(self.__select_value,
                                        Features=self.__features_w,
                                        Values=self.__values_w,
                                        Color_Picker=self.__color_picker_w)

            self.__full_widgets_ui.children = new_i.children