from eflow._hidden.parent_objects import JupyterWidget

import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Layout


class StringCleaningWidget(JupyterWidget):

    def __init__(self):

        JupyterWidget.__init__(self,
                               self.__class__.__name__)

        # Define needed widgets
        self.__feature_cleaning_options_w = None
        self.__features_w = None
        self.__options_w = None
        self.__input_w = None
        self.__full_widgets_ui = None

        # ---
        self.__require_input = dict()

    def run_widget(self,
                   df,
                   df_features):
        """
        Desc:

        Args:
            df: pd.Dataframe
                A pandas dataframe object

            df_features: DataFrameType
                DataFrameTypes object; organizes feature types into groups.

        Returns/Descr:
            Returns a UI widget to create a JSON file for cleaning.
        """
        self.__feature_option_dict = dict()

        self.__selected_options = {feature_name: "None"
                                   for feature_name in
                                   df_features.string_features()}

        feature_cleaning_options = {feature_name: ["None"]
                                    for feature_name in
                                    df_features.string_features()}

        self.__feature_options_w = {key: widgets.Select(
            options=feature_cleaning_options[key],
            layout=Layout(width='50%',
                          height='300px'))
            for key in feature_cleaning_options}

        self.__features_w = widgets.Select(
            options=list(feature_cleaning_options.keys()),
            layout=Layout(width='50%',
                          height='175px')
        )

        self.__update_widgets()

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
        self.__submit_button = widgets.Button(
            description='Create JSON File from options',
            color="#ff1122",
            layout=Layout(left='100px',
                          bottom="5px",
                          width='40%', ))

        self.__input_w = widgets.Text(
            value='',
            placeholder='Replace Value',
            description='Input:',
            disabled=False,
            visible=False,
            layout=Layout(width='50%')
        )

        # Link functions with non-updateable widgets
        self.__features_w.observe(self.__select_feature,
                                  'value')

        self.__input_w.observe(self.__validate_save_input_w)

        # Setup and display full UI
        self.__full_widgets_ui = widgets.interactive(
            self.__save_option,
            Features=self.__features_w,
            Options=self.__options_w,
            Text_Input=self.__input_w,
        )

        display(self.__full_widgets_ui)

    def __update_widgets(self):
        init = self.__features_w.value
        self.__options_w = self.__feature_options_w[init]

    def __validate_save_input_w(self,
                                _):
        """
        Returns/Descr:
            Ensures the input field is within specified parameters defined
            by the 'require_input' dictionary.
        """

        pass

    def __hide_init_input_w(self,
                            _):

        if self.__options_w.value in self.__require_input:
            self.__input_w.layout.visibility = 'visible'
        else:
            self.__input_w.layout.visibility = 'hidden'
            self.__input_w.layout.visibility = 'visible'
            # self.__input_w.value = ""

    def __save_option(self,
                      **func_kwargs):
        self.__selected_options[
            func_kwargs["Features"]] = self.__options_w.value
        self.__hide_init_input_w(None)

    def __select_feature(self,
                         feature):
        """
        Returns/Descr:
            When a feature selection is chosen all the widgets are
            re-initialized.
        """

        self.__update_widgets()

        new_i = widgets.interactive(self.__save_option,
                                    Features=self.__features_w,
                                    Options=self.__options_w,
                                    Text_Input=self.__input_w)

        self.__full_widgets_ui.children = new_i.children

# StringCleaningWidget().run_widget(df,
#                                   df_features)