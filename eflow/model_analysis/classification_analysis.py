from eflow.utils.sys_utils import *
from eflow.utils.pandas_utils import df_to_image
from eflow.utils.image_processing_utils import create_plt_png
from eflow._hidden.parent_objects import FileOutput
from eflow._hidden.custom_exceptions import RequiresPredictionMethods, ProbasNotPossible, \
    UnsatisfiedRequirments
from eflow.data_analysis import FeatureAnalysis

from eflow._hidden.constants import GRAPH_DEFAULTS

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import scikitplot as skplt
import numpy as np
import warnings
import copy
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class ClassificationAnalysis(FileOutput):

    """
        Analyzes a classification model's result's based on the prediction
        function(s) passed to it. Creates graphs and tables to be saved in directory
        structure.
    """

    def __init__(self,
                 model,
                 model_name,
                 target_feature,
                 pred_funcs_dict,
                 df_features,
                 sample_data,
                 project_sub_dir="Classification Analysis",
                 overwrite_full_path=None,
                 target_classes=None,
                 save_model=True):
        """
        Args:
            model:
                A fitted supervised machine learning model.

            model_name:
                The name of the model in string form.

            pred_funcs_dict:
                A dict of the name of the function and the function defintion for the
                model prediction methods.
                (Can handle either a return of probabilities or a singile value.)
                Init Example:
                pred_funcs = dict()
                pred_funcs["Predictions"] = model.predict
                pred_funcs["Probabilities"] = model.probas

            sample_data:
                Given data to then pass into our prediction functions to get a
                resultant to get the classification prediction 'type'.
                Can be a matrix or a vector.

            project_sub_dir:
                Creates a parent or "project" folder in which all sub-directories
                will be inner nested.

            overwrite_full_path:
                Overwrites the path to the parent folder.

            target_classes:
                Specfied list/np.array of targeted classes the model predicts. If set to
                none then it will attempt to pull from the sklearn's default attribute
                '.classes_'.

            df_features:
                DataFrameTypes object; organizes feature types into groups.
        """

        # Init parent object
        FileOutput.__init__(self,
                            f'{project_sub_dir}/Target Feature: {target_feature}/{model_name}',
                            overwrite_full_path)

        # Init objects without pass by refrence
        self.__model = copy.deepcopy(model)
        self.__model_name = copy.deepcopy(model_name)
        self.__target_values = copy.deepcopy(target_classes)
        self.__df_features = copy.deepcopy(df_features)
        self.__pred_funcs_dict = copy.deepcopy(pred_funcs_dict)
        self.__pred_funcs_types = dict()

        # Init on sklearns default target classes attribute
        if not self.__target_values:
            self.__target_values = copy.deepcopy(model.classes_)
        # ---
        if len(self.__target_values) != 2:
            self.__binary_classifcation = False
        else:
            self.__binary_classifcation = True

        # Save machine learning model
        if save_model:
            pickle_object_to_file(self.__model,
                                  self.folder_path,
                                  f'{self.__model_name}')

        # ---
        create_dir_structure(self.folder_path,
                             "_Extras")
        # Save predicted classes
        write_object_text_to_file(self.__target_values,
                                  self.folder_path + "_Extras",
                                  "_Classes")

        # Save features and or df_features object
        df_features.create_json_file_representation(self.folder_path + "_Extras",
                                                    "df_features.json")

        self.__sample_data = None

        if len(sample_data.shape) == 2:
            self.__sample_data = np.reshape(sample_data[0],
                                            (-1, sample_data.shape[1]))
        elif len(sample_data.shape) == 1:
            self.__sample_data = [sample_data]
        else:
            raise UnsatisfiedRequirments("This program can only handle 1D and 2D matrices.")

        print(self.__sample_data)

        # Find the 'type' of each prediction. Probabilities or Predictions
        if self.__pred_funcs_dict:
            for pred_name, pred_func in self.__pred_funcs_dict.items():
                model_output = pred_func(self.__sample_data)[0]

                # Confidence / Probability (Continuous output)
                if isinstance(model_output, list) or isinstance(model_output,
                                                                np.ndarray):
                    self.__pred_funcs_types[pred_name] = "Probabilities"

                # Classification (Discrete output)
                else:
                    self.__pred_funcs_types[pred_name] = "Predictions"
        else:
            raise RequiresPredictionMethods


    def perform_analysis(self,
                         X,
                         y,
                         dataset_name,
                         thresholds_matrix=None,
                         figsize=GRAPH_DEFAULTS.FIGSIZE,
                         ignore_metrics=[],
                         custom_metrics_dict=dict(),
                         average_scoring=["micro",
                                          "macro",
                                          "weighted"],
                         display_visuals=False):
        """
        Desc:
            Runs all available analysis functions on the models predicted data.

        Args:
            X:
                Feature matrix.

            y:
                Target data vector.

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds_matrix:
                List of list/matrix of thresholds.

                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            figsize:
                All plot's dimension's.

            ignore_metrics:
                Specify the default metrics to not apply to the classification
                data_analysis.
                    * Precision
                    * MCC
                    * Recall
                    * F1-Score
                    * Accuracy

            custom_metrics_dict:
                Pass the name of metric(s) with the function definition(s) in a
                dictionary.

            average_scoring:
                Determines the type of averaging performed on the data.

            display_visuals:
                Controls visual display of error error data_analysis if it is able to run.

        Returns:
            Performs all classification functionality with the provided feature
            data and target data.
                * plot_precision_recall_curve
                * classification_evaluation
                * plot_confusion_matrix
        """

        # Convert to
        if isinstance(thresholds_matrix, np.ndarray):
            thresholds_matrix = thresholds_matrix.tolist()

        if not thresholds_matrix:
            thresholds_matrix = list()

        if isinstance(thresholds_matrix, list) and not isinstance(
                thresholds_matrix[0], list):
            thresholds_matrix = list(thresholds_matrix)

        if None not in thresholds_matrix:
            thresholds_matrix.append(None)

        print("\n\n" + "---" * 10 + f'{dataset_name}' + "---" * 10)
        first_iteration = True
        for pred_name, pred_type in self.__pred_funcs_types.items():

            # Nicer formating
            if not first_iteration:
                print("\n\n\n")
            first_iteration = False

            for thresholds in thresholds_matrix:


                print(f"Now running classification on {pred_name}", end='')
                if pred_type == "Predictions":
                    print()
                    thresholds = None
                else:
                    if thresholds:
                        print(f" on thresholds:\n{thresholds}")
                    else:
                        print(" on no thresholds.")
                print("-" * (len(dataset_name) + 60))

                self.classification_metrics(X,
                                            y,
                                            pred_name=pred_name,
                                            dataset_name=dataset_name,
                                            thresholds=thresholds,
                                            ignore_metrics=ignore_metrics,
                                            custom_metrics_dict=custom_metrics_dict,
                                            average_scoring=average_scoring,
                                            display_visuals=display_visuals)

                self.plot_confusion_matrix(X,
                                           y,
                                           pred_name=pred_name,
                                           dataset_name=dataset_name,
                                           thresholds=thresholds,
                                           figsize=figsize,
                                           normalize=True,
                                           display_visuals=display_visuals)

                self.plot_confusion_matrix(X,
                                           y,
                                           pred_name=pred_name,
                                           dataset_name=dataset_name,
                                           thresholds=thresholds,
                                           figsize=figsize,
                                           normalize=False,
                                           display_visuals=display_visuals)

                if pred_type == "Probabilities":
                    self.plot_precision_recall_curve(X,
                                                     y,
                                                     pred_name=pred_name,
                                                     dataset_name=dataset_name,
                                                     figsize=figsize,
                                                     thresholds=thresholds,
                                                     display_visuals=display_visuals)
                    self.plot_roc_curve(X,
                                        y,
                                        pred_name=pred_name,
                                        dataset_name=dataset_name,
                                        figsize=figsize,
                                        thresholds=thresholds,
                                        display_visuals=display_visuals)

                    if self.__binary_classifcation:

                        self.plot_lift_curve(X,
                                             y,
                                             pred_name=pred_name,
                                             dataset_name=dataset_name,
                                             figsize=figsize,
                                             thresholds=thresholds,
                                             display_visuals=display_visuals)

                        self.plot_ks_statistic(X,
                                               y,
                                               pred_name=pred_name,
                                               dataset_name=dataset_name,
                                               figsize=figsize,
                                               thresholds=thresholds,
                                               display_visuals=display_visuals)

                        self.plot_cumulative_gain(X,
                                                  y,
                                                  pred_name=pred_name,
                                                  dataset_name=dataset_name,
                                                  figsize=figsize,
                                                  thresholds=thresholds,
                                                  display_visuals=display_visuals)

                # if self.__df_features:
                #     self.classification_error_analysis(X,
                #                                        y,
                #                                        pred_name=pred_name,
                #                                        dataset_name=dataset_name,
                #                                        thresholds=thresholds,
                #                                        display_analysis_graphs=display_analysis_graphs)

                print("-" * (len(dataset_name) + 60))
                if pred_type == "Predictions":
                    break

    def plot_ks_statistic(self,
                          X,
                          y,
                          pred_name,
                          dataset_name,
                          thresholds=None,
                          display_visuals=True,
                          save_file=True,
                          title=None,
                          ax=None,
                          figsize=None,
                          title_fontsize='large',
                          text_fontsize='medium'):

        """
        Desc:
            From scikit-plot documentation
            Link: http://tinyurl.com/y3ym5pyc
            Generates the KS Statistic plot from labels and scores/probabilities.

        Args:
            X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """

        filename = f'KS Statistic on {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)
        if not title:
            title = filename

        skplt.metrics.plot_ks_statistic(y,
                                        self.__get_model_probas(pred_name,
                                                                X),
                                        title=title,
                                        ax=ax,
                                        figsize=figsize,
                                        title_fontsize=title_fontsize,
                                        text_fontsize=text_fontsize)
        if save_file:
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if display_visuals:
            plt.show()
        plt.close()

    def plot_roc_curve(self,
                       X,
                       y,
                       pred_name,
                       dataset_name,
                       thresholds=None,
                       display_visuals=True,
                       save_file=True,
                       title=None,
                       ax=None,
                       figsize=GRAPH_DEFAULTS.FIGSIZE,
                       title_fontsize='large',
                       text_fontsize='medium'):

        """
        Desc:
            From scikit-plot documentation
            Link: http://tinyurl.com/y3ym5pyc
            Creates ROC curves from labels and predicted probabilities.

        Args:
             X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """
        filename = f'Roc Curve on {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)
        if not title:
            title = filename

        skplt.metrics.plot_roc(y,
                               self.__get_model_probas(pred_name,
                                                       X),
                               title=title,
                               ax=ax,
                               figsize=figsize,
                               title_fontsize=title_fontsize,
                               text_fontsize=text_fontsize)

        if save_file:
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if display_visuals:
            plt.show()
        plt.close()

    def plot_cumulative_gain(self,
                             X,
                             y,
                             pred_name,
                             dataset_name,
                             thresholds=None,
                             display_visuals=True,
                             save_file=True,
                             title=None,
                             ax=None,
                             figsize=GRAPH_DEFAULTS.FIGSIZE,
                             title_fontsize='large',
                             text_fontsize='medium'):

        """
        Desc:
            From scikit-plot documentation
            Link: http://tinyurl.com/y3ym5pyc
            Plots calibration curves for a set of classifier probability estimates.

        Args:
            X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """
        filename = f'Cumulative Gain gain on {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)
        if not title:
            title = filename

        skplt.metrics.plot_cumulative_gain(y,
                                           self.__get_model_probas(pred_name,
                                                               X),
                                           title=title,
                                           ax=ax,
                                           figsize=figsize,
                                           title_fontsize=title_fontsize,
                                           text_fontsize=text_fontsize)

        if save_file:
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if display_visuals:
            plt.show()
        plt.close()

    def plot_precision_recall_curve(self,
                                    X,
                                    y,
                                    pred_name,
                                    dataset_name,
                                    thresholds=None,
                                    display_visuals=True,
                                    save_file=True,
                                    title=None,
                                    plot_micro=True,
                                    classes_to_plot=None,
                                    ax=None,
                                    figsize=GRAPH_DEFAULTS.FIGSIZE,
                                    cmap='nipy_spectral',
                                    title_fontsize='large',
                                    text_fontsize='medium'):
        """
        Desc:
            From scikit-plot documentation
            Link: http://tinyurl.com/y3ym5pyc
            Plots precision recall curve plot based on the models predictions.

        Args:
            X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """

        filename = f'Precision Recall on {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)
        if not title:
            title = filename

        skplt.metrics.plot_precision_recall(y,
                                            self.__get_model_probas(pred_name,
                                                                    X),
                                            title=title,
                                            plot_micro=plot_micro,
                                            classes_to_plot=classes_to_plot,
                                            ax=ax,
                                            figsize=figsize,
                                            cmap=cmap,
                                            title_fontsize=title_fontsize,
                                            text_fontsize=text_fontsize)

        if save_file:
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if display_visuals:
            plt.show()
        plt.close()

    def plot_lift_curve(self,
                        X,
                        y,
                        pred_name,
                        dataset_name,
                        thresholds=None,
                        display_visuals=True,
                        save_file=True,
                        title=None,
                        ax=None,
                        figsize=GRAPH_DEFAULTS.FIGSIZE,
                        title_fontsize='large',
                        text_fontsize='medium'):
        """
        Desc:
            From scikit-plot documentation
            Link: http://tinyurl.com/y3ym5pyc
            The lift curve is used to determine the effectiveness of a binary classifier.
            A detailed explanation can be found at http://tinyurl.com/csegj9.
            The implementation here works only for binary classification.

        Args:
            X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """

        filename = f'Lift Curve on {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)
        if not title:
            title = filename

        skplt.metrics.plot_lift_curve(y,
                                      self.__get_model_probas(pred_name,
                                                              X),
                                      title=title,
                                      ax=ax,
                                      figsize=figsize,
                                      title_fontsize=title_fontsize,
                                      text_fontsize=text_fontsize)

        if save_file:
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if display_visuals:
            plt.show()
        plt.close()

    def plot_confusion_matrix(self,
                              X,
                              y,
                              pred_name,
                              dataset_name,
                              thresholds=None,
                              display_visuals=True,
                              save_file=True,
                              title=None,
                              normalize=False,
                              hide_zeros=False,
                              hide_counts=False,
                              x_tick_rotation=0,
                              ax=None,
                              figsize=GRAPH_DEFAULTS.FIGSIZE,
                              cmap='Blues',
                              title_fontsize='large',
                              text_fontsize='medium'):
        """
        Desc:
            From scikit-plot documentation
            Link: http://tinyurl.com/y3ym5pyc
            Creates a confusion matrix plot based on the models predictions.

        Args:
            X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """
        filename = f'Confusion Matrix: {dataset_name} on {self.__model_name} Normalized: {normalize}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)
        if not title:
            title = filename

        warnings.filterwarnings('ignore')
        ax = skplt.metrics.plot_confusion_matrix(
             self.__get_model_prediction(pred_name,
                                         X,
                                         thresholds),
             y,
             title=title,
             normalize=normalize,
             hide_zeros=hide_zeros,
             hide_counts=hide_counts,
             x_tick_rotation=x_tick_rotation,
             ax=ax,
             figsize=figsize,
             cmap=cmap,
             title_fontsize=title_fontsize,
             text_fontsize=text_fontsize)

        bottom, top = ax.get_ylim()
        ax.set_ylim(bottom + 0.5, top - 0.5)

        warnings.filterwarnings('default')

        if save_file:
            create_plt_png(self.folder_path,
                           sub_dir,
                           convert_to_filename(filename))

        if display_visuals:
            plt.show()
        plt.close()

    def classification_metrics(self,
                               X,
                               y,
                               pred_name,
                               dataset_name,
                               thresholds=None,
                               display_visuals=True,
                               save_file=True,
                               title="",
                               custom_metrics_dict=dict(),
                               ignore_metrics=[],
                               average_scoring=["micro",
                                                "macro",
                                                "weighted"]):
        """
        Desc:
            Creates a dataframe based on the prediction metrics
            of the feature matrix and target vector.

        Args:
            X:
                Feature matrix.

            y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned stored
                in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Display tables.

            save_file:
                Determines whether or not to save the generated document.

            title:
                Adds to the column 'Metric Score'.

            custom_metrics_dict:
                Pass the name of metric(s) and the function definition(s) in a
                dictionary.

            ignore_metrics:
                Specify the default metrics to not apply to the classification
                data_analysis.
                    * Precision
                    * MCC
                    * Recall
                    * F1-Score
                    * Accuracy

            average_scoring:
                Determines the type of averaging performed on the data.
                    * micro
                    * macro
                    * weighted

        Returns:
            Return a dataframe object of the metrics value.
        """
        filename = f'Metric Evaluation on {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        if not isinstance(average_scoring, list):
            average_scoring = [average_scoring]

        # Default metric name's and their function
        metric_functions = dict()
        metric_functions["Precision"] = precision_score
        metric_functions["MCC"] = matthews_corrcoef
        metric_functions["Recall"] = recall_score
        metric_functions["F1-Score"] = f1_score
        metric_functions["Accuracy"] = accuracy_score

        warnings.filterwarnings('ignore')

        # Ignore default metrics if needed
        for remove_metric in ignore_metrics:
            if remove_metric in metric_functions:
                del metric_functions[remove_metric]

        # Add in custom metrics
        if len(custom_metrics_dict.keys()):
            metric_functions.update(custom_metrics_dict)

        # Evaluate model on metrics
        evaluation_report = dict()
        for metric_name in metric_functions:
            for average_score in average_scoring:

                model_predictions = self.__get_model_prediction(pred_name,
                                                                X,
                                                                thresholds)
                try:
                    evaluation_report[f'{metric_name}({average_score})'] = \
                        metric_functions[metric_name](y_true=y,
                                                      y_pred=model_predictions,
                                                      average=average_score)
                except TypeError:
                    evaluation_report[metric_name] = metric_functions[
                        metric_name](y,
                                     model_predictions)
                    break

        warnings.filterwarnings('default')

        if title and len(title) > 0:
            index_name = f"Metric Scores ({title})"
        else:
            index_name = "Metric Scores"

        # ---
        evaluation_report = pd.DataFrame({index_name:
                                              [f'{metric_score:.4f}'
                                               for metric_score
                                               in evaluation_report.values()]},
                                         index=list(evaluation_report.keys()))

        if display_visuals:
            display(evaluation_report)
        else:
            print(evaluation_report)

        if save_file:
            df_to_image(evaluation_report,
                        self.folder_path,
                        sub_dir,
                        convert_to_filename(filename),
                        col_width=20,
                        show_index=True,
                        format_float_pos=4)

    def classification_error_analysis(self,
                                      X,
                                      y,
                                      pred_name,
                                      dataset_name,
                                      thresholds=None,
                                      display_visuals=True,
                                      save_file=True,
                                      display_analysis_graphs=False):
        """
        Args:
            X:
                Feature matrix.

            y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            save_file:
                Determines whether or not to save the generated document.

            display_analysis_graphs:
                Controls visual display of graph generation.
        """
        pass
        # sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
        #                                                 dataset_name,
        #                                                 thresholds)
        #
        # model_predictions = self.__get_model_prediction(pred_name,
        #                                                 X,
        #                                                 thresholds=thresholds)
        #
        # if sum(model_predictions == y):
        #     if display_analysis_graphs:
        #         print("\n\n" + "*" * 10 +
        #               "Correctly predicted data_analysis"
        #               + "*" * 10 + "\n")
        #     else:
        #         print("\n\n" + "*" * 10 +
        #               "Generating graphs for model's correctly predicted..." +
        #               "*" * 10 + "\n")
        #         FeatureAnalysis(pd.DataFrame(X[model_predictions == y],
        #                                   columns=self.__df_features.get_all_features()),
        #                      self.__df_features,
        #                      overwrite_full_path=self.folder_path +
        #                                          sub_dir + "/Correctly Predicted Data/",
        #                      notebook_mode=display_analysis_graphs)
        # else:
        #     print("Your model predicted nothing correctly...dam that sucks")
        #
        # if sum(model_predictions != y):
        #     if display_analysis_graphs:
        #         print("\n\n" + "*" * 10 +
        #               "Incorrectly predicted data_analysis"
        #               + "*" * 10 + "\n")
        #     else:
        #         print("\n\n" + "*" * 10 +
        #               "Generating graphs for model's incorrectly predicted..." +
        #               "*" * 10 + "\n")
        #
        #     DataAnalysis(pd.DataFrame(X[model_predictions != y],
        #                               columns=self.__df_features.get_all_features()),
        #                  self.__df_features,
        #                  overwrite_full_path=self.folder_path +
        #                                      sub_dir + "/Incorrectly Predicted Data/",
        #                  notebook_mode=display_analysis_graphs)
        # else:
        #     print(
        #         "\n\nYour model predicted everything correctly...there is something very wrong here...")

    def classification_report(self,
                              X,
                              y,
                              pred_name,
                              dataset_name,
                              thresholds=None,
                              display_visuals=True,
                              save_file=True):
        """
        Desc:
            Creates a report of all target's metric evaluations
            based on the model's prediction output from the classification report
            from the sklearn.

        Args:
            X:
                Feature matrix.

             y:
                Target data vector.

            pred_name:
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals:
                Visualize graph if needed.

            save_file:
                Boolean value to whether or not to save the file.
        """
        filename = f'Classification Report {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        # Create dataframe report
        report_df = pd.DataFrame(classification_report(y,
                                                       self.__get_model_prediction(
                                                           pred_name,
                                                           X,
                                                           thresholds),
                                                       output_dict=True))

        # ---
        if display_visuals:
            display(report_df)
        else:
            print(report_df)

        if save_file:
            # Output dataframe as png
            df_to_image(report_df,
                        self.folder_path,
                        sub_dir,
                        filename,
                        col_width=20,
                        show_index=True,
                        format_float_pos=4)

    def __get_model_prediction(self,
                               pred_name,
                               X,
                               thresholds=None):
        """
        Desc:
            Finds the model's predicted labels.

        Args:
            X:
                Feature matrix.

            pred_name:
                The name of the prediction function in questioned stored in 'self.__pred_funcs_dict'

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

        Returns:
            Returns back a predicted value based for a given matrix.
            Handles prediction function 'types' Predictions and Probabilities.
            Helps streamline the entire process of evaluating classes.
        """

        # Check if function name exists in the dictionary
        if pred_name not in self.__pred_funcs_types:
            raise KeyError("The function name has not be recorded!")

        # Must be a prediction function
        if self.__pred_funcs_types[pred_name] == "Predictions":
            return self.__pred_funcs_dict[pred_name](X)

        # Output must be continuous; Probabilities
        elif self.__pred_funcs_types[pred_name] == "Probabilities":

            # Validate probabilities
            if thresholds:
                if isinstance(thresholds, list) or \
                        isinstance(thresholds, np.ndarray):
                    if len(thresholds) != len(self.__target_values):
                        raise UnsatisfiedRequirments("Length of thresholds must match the same length as the associated classes.")
                else:
                    raise UnsatisfiedRequirments("Threshold object is not a list or numpy array!")

            # Get model probability output
            model_output = self.__get_model_probas(pred_name,
                                                   X)

            # No thresholds found
            if not thresholds:
                return np.asarray([self.__target_values[np.argmax(proba)]
                                   for proba in model_output])

            # Apply thresholds to model's probability output
            else:
                model_output = model_output - thresholds

                return np.asarray([self.__target_values[np.argmax(proba)]
                                   for proba in model_output])
        else:
            raise ValueError(f"Unknown type '{self.__pred_funcs_types[pred_name]}' was found!")

    def __get_model_probas(self,
                           pred_name,
                           X):
        """
        Desc:
            Attempts to get the probabilities from the prediction function.

        Args:
            X:
                Feature matrix.

            pred_name:
                The name of the prediction function in questioned stored in 'self.__pred_funcs_dict'

        Raises:
             If probabilities isn't possible with the given function that it will invoke an error.

        Returns:
            Returns back a series of values between 0-1 to represent it's confidence.
        """

        if self.__pred_funcs_types[pred_name] == "Probabilities":
            model_output = self.__pred_funcs_dict[pred_name](X)

            # ---
            if isinstance(model_output, list):
                model_output = np.asarray(model_output)

            return model_output
        else:
            raise ProbasNotPossible

    def __create_sub_dir_with_thresholds(self,
                                         pred_name,
                                         dataset_name,
                                         thresholds):
        """
        Desc:
            Iterates through directory structure looking at each text file
            containing a string representation of the given threshold;
            keeps comparing the passed value of 'thresholds' to the text file.

        Args:
            pred_name:
                The name of the prediction function in questioned stored in 'self.__pred_funcs_dict'

            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

        Returns:
            Looking at the root of the starting directory and looking at each
            '_Thresholds.txt' file to determine if the files can be outputed
            to that directory. The content of the file must match the content
            of the list/numpy array 'thresholds'.
        """

        sub_dir = f'{dataset_name}/{pred_name}'

        # Only generate extra folder structure if function type is Probabilities
        if self.__pred_funcs_types[pred_name] == "Probabilities":

            # ------
            if not thresholds:
                sub_dir = f'{sub_dir}/No Thresholds'
            else:
                i = 0
                sub_dir = f'{sub_dir}/Thresholds'
                tmp_sub_dir = copy.deepcopy(sub_dir)
                while True:
                    threshold_dir = self.folder_path
                    if i > 0:
                        tmp_sub_dir = (sub_dir + f' {i}')
                    threshold_dir += tmp_sub_dir

                    # If file exists with the same thresholds; than use this directory
                    if os.path.exists(threshold_dir):
                        if self.__compare_thresholds_to_saved_thresholds(
                                threshold_dir,
                                thresholds):
                            sub_dir = tmp_sub_dir
                            break

                    # Create new directory
                    else:
                        os.makedirs(threshold_dir)
                        write_object_text_to_file(thresholds,
                                                  threshold_dir,
                                                  "_Thresholds")
                        sub_dir = tmp_sub_dir
                        break

                    # Iterate for directory name change
                    i += 1

        return sub_dir

    def __compare_thresholds_to_saved_thresholds(self,
                                                 directory_path,
                                                 thresholds):
        """
        Desc:
            Compare the thresholds object to a threshold text file found in
            the directory; returns true if the file exists and the object's
            value matches up.

        Args:
            directory_path:
                Path to the given folder where the "_Thresholds.txt"

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

        Returns:
            Compare the thresholds object to the text file; returns true if
            the file exists and the object's value matches up.
        """

        file_directory = correct_directory_path(directory_path)

        if os.path.exists(file_directory):

            # Extract file contents and convert to a list object
            file = open(file_directory + "_Thresholds.txt", "r")
            line = file.read()
            converted_list = line.split("=")[-1].strip().strip('][').split(
                ', ')
            converted_list = [float(val) for val in converted_list]
            file.close()

            if thresholds == converted_list:
                return True
            else:
                return False
        else:
            return False