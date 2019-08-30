from eFlow._Hidden.Objects.enum import enum
from eFlow.Utils.SysUtils import create_plt_png, convert_to_file_name, df_to_image
from eFlow._Hidden.Objects.FileOutput import *
from eFlow._Hidden.Constants import PREDICTION_TYPES
from eFlow._Hidden.CustomExc import *
from eFlow.Analysis.DataAnalysis import *

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import scikitplot as skplt
import numpy as np
import warnings
import copy
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt

class SupervisedModelAnalysis(FileOutput):

    def __init__(self,
                 model,
                 model_name,
                 pred_func,
                 X_train,
                 y_train,
                 X_test=None,
                 y_test=None,
                 X_val=None,
                 y_val=None,
                 prediction_type=PREDICTION_TYPES.CLASSIFICATION,
                 project_name="Supervised Analysis",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 thresholds=None,
                 overwrite_target_classes=None,
                 df_features=None):
        """
        model:
            A fitted supervised machine learning model.

        model_name:
            The name of the model.

        X_train/y_train | X_test/y_test | X_val/y_val:
            Feature matrix/Target data.

        prediction_type:
            Must be a 'Classification' or 'Regression' based model

        project_name:
            Creates a parent or "project" folder in which all sub-directories
            will be inner nested.

        overwrite_full_path:
            Overwrites the path to the parent folder.

        notebook_mode:
            If in a python notebook display in the notebook.

        Returns/Desc:
            Creates plots and dataframes for display and saves them in a
            directory path.
        """

        # Init any parent objects
        FileOutput.__init__(self,
                            f'{project_name}/{model_name}',
                            overwrite_full_path)

        print(self.get_output_folder())
        # Init objectss by pass by refrence
        self.__model = copy.deepcopy(model)
        self.__model_name = copy.deepcopy(model_name)
        self.__notebook_mode = copy.deepcopy(notebook_mode)
        self.__prediction_type = copy.deepcopy(prediction_type)
        self.__target_values = None

        self.__pred_func = None
        self.__proba_func = None
        self.__df_features = copy.deepcopy(df_features)

        if pred_func:
            model_output = pred_func(
                np.reshape(X_train[0],
                           (-1, X_train.shape[1])))[0]
            # Must be a confidence probability output (continuous values)

            if isinstance(model_output, list) or isinstance(model_output,
                                                            np.ndarray):
                self.__proba_func = copy.deepcopy(pred_func)

            # Regression or Classification
            else:
                self.__pred_func = copy.deepcopy(pred_func)


        # Classification model
        if prediction_type == PREDICTION_TYPES.CLASSIFICATION:

            if not overwrite_target_classes:
                self.__target_values = model.classes_
            else:
                self.__target_values = overwrite_target_classes

            if len(self.__target_values) == 2:
                self.__binary_classification = True
            else:
                self.__binary_classification = False

            if X_train is not None and y_train is not None:
                self.classification_analysis(X_train,
                                             y_train,
                                             dataset_name="Train data",
                                             thresholds=thresholds)

            if X_test is not None and y_test is not None:
                self.classification_analysis(X_test,
                                             y_test,
                                             dataset_name="Test data",
                                             thresholds=thresholds)

            if X_val is not None and y_val is not None:
                self.classification_analysis(X_val,
                                             y_val,
                                             dataset_name="Validation data",
                                             thresholds=thresholds)

        # Regression model
        elif prediction_type == PREDICTION_TYPES.REGRESSION:
            pass
        else:
            raise UnknownPredictionType

    def get_model_prediction(self,
                             X,
                             thresholds=None):
        """
        X:
            Feature matrix.

        Thresholds:
            If the model outputs a probability list/numpy array then we apply
            thresholds to each classification.

        Returns/Desc:
            Returns back a predicted value based for a given matrix.
        """

        if self.__pred_func:
            return self.__pred_func(X)
        else:
            if thresholds:
                if isinstance(thresholds, list) or \
                        isinstance(thresholds, np.ndarray):
                    if sum(thresholds) != 1:
                        print("Thresholds didn't add up to 100%! "
                              "This may cause issues in your results!")
                        pass
                    if len(thresholds) != len(self.__target_values):
                        raise ThresholdLength
                else:
                    raise ThresholdType

            model_output = self.get_model_probas(X)
            if isinstance(model_output, list):
                model_output = np.asarray(model_output)

            if isinstance(model_output, np.ndarray):
                if thresholds:
                    model_output = model_output - np.asarray(thresholds)
                return np.asarray([self.__target_values[np.argmax(proba)]
                                   for proba in model_output])
            raise UnknownModelOutputType

    def get_model_probas(self,
                         X):
        if self.__proba_func:
            return self.__proba_func(X)
        else:
            raise ProbasNotPossible

    def classification_analysis(self,
                                X,
                                y,
                                dataset_name,
                                normalize_confusion_matrix=True,
                                ignore_metrics=[],
                                custom_metrics=dict(),
                                average_scoring=["micro",
                                                 "macro",
                                                 "weighted"],
                                thresholds=None,
                                display_graphs=False):
        """
        X/y:
            Feature matrix/Target data vector

        dataset_name:
            The dataset's name

        normalize_confusion_matrix:
            Normalize the confusion matrix buckets.

        ignore_metrics:
            Specify set metrics to ignore. (F1-Score, Accuracy etc)

        ignore_metrics:
            Specify the default metrics to not apply to the classification
            analysis.
                * Precision
                * MCC
                * Recall
                * F1-Score
                * Accuracy

        average_scoring:
            Determines the type of averaging performed on the data.

        Returns/Desc:
            Performs all classification functionality with the provided feature
            data and target data.
                * plot_precision_recall_curve
                * classification_evaluation
                * plot_confusion_matrix
        """

        if self.__prediction_type == PREDICTION_TYPES.CLASSIFICATION:



            print("\n\n" + "---" * 10 + f'{dataset_name}' + "---" * 10)

            if self.__pred_func:
                predict_dir = "Prediction Classification"
            elif self.__proba_func:
                predict_dir = "Probability Classification"
            else:
                predict_dir = "Unknown Classification Type"


            if self.__proba_func:
                tmp_file_name = f'Precision Recall Curve with Scores ' + \
                                f'on {dataset_name}'
                self.plot_precision_recall_curve(X,
                                                 y,
                                                 sub_dir=f'{dataset_name}/{predict_dir}',
                                                 title=tmp_file_name,
                                                 filename=tmp_file_name,
                                                 thresholds=thresholds)

            self.classification_metrics(X,
                                        y,
                                        title=dataset_name,
                                        sub_dir=f'{dataset_name}/{predict_dir}',
                                        ignore_metrics=ignore_metrics,
                                        custom_metrics=custom_metrics,
                                        file_name=f'Evaluation on ' + \
                                        f'{dataset_name}',
                                        average_scoring=average_scoring)

            tmp_file_name = f'Confusion Matrix: {dataset_name}'
            self.plot_confusion_matrix(X,
                                       y,
                                       sub_dir=f'{dataset_name}/{predict_dir}',
                                       title=tmp_file_name,
                                       normalize=normalize_confusion_matrix,
                                       file_name=tmp_file_name)

            if self.__binary_classification:
                skplt.metrics.plot_ks_statistic(y,
                                                self.__model.predict_proba(X),
                                                figsize=(10, 8))
            if self.__df_features:
                self.classification_error_analysis(X,
                                                   y,
                                                   sub_dir=f'{dataset_name}/{predict_dir}',
                                                   thresholds=thresholds,
                                                   display_graphs=display_graphs)

        else:
            print(f'Model prediction is not a classification problem.' + \
                  f'It is a {self.__prediction_type}')


    def plot_precision_recall_curve(self,
                                    X,
                                    y,
                                    sub_dir="",
                                    figsize=(10, 8),
                                    title=None,
                                    filename=None,
                                    thresholds=None):
        """
        X/y:
            Feature matrix/Target data vector.

        sub_dir:
            Specify a subdirectory to append to the output path of the file.

        figsize:
            Plot's size.

        title:
            Title of the plot.

        filename:
            Name of the file. If set to 'None' then don't create the file.

        Returns/Desc:

        """
        if title:
            skplt.metrics.plot_precision_recall(self.get_model_probas(X),
                                                y,
                                                figsize=figsize,
                                                title=title)
        else:
            skplt.metrics.plot_precision_recall(self.get_model_probas(X),
                                                y,
                                                figsize=figsize)

        if filename:
            create_plt_png(self.get_output_folder(),
                           sub_dir,
                           convert_to_file_name(filename))

        if self.__notebook_mode:
            plt.show()
        plt.close()

    def plot_confusion_matrix(self,
                              X,
                              y,
                              sub_dir="",
                              figsize=(10, 8),
                              title=None,
                              file_name=None,
                              normalize=True,
                              thresholds=None,
                              ):

        warnings.filterwarnings('ignore')
        if title:
            skplt.metrics.plot_confusion_matrix(self.get_model_prediction(X,
                                                                          thresholds),
                                                y,
                                                figsize=figsize,
                                                title=title,
                                                normalize=normalize,)
        else:
            skplt.metrics.plot_confusion_matrix(self.get_model_prediction(X,
                                                                          thresholds),
                                                y,
                                                figsize=figsize,
                                                normalize=normalize)
        warnings.filterwarnings('default')

        if file_name:
            create_plt_png(self.get_output_folder(),
                           sub_dir,
                           convert_to_file_name(file_name))

        if self.__notebook_mode:
            plt.show()
        plt.close()

    def classification_metrics(self,
                               X,
                               y,
                               title="",
                               sub_dir="",
                               file_name=None,
                               custom_metrics=dict(),
                               ignore_metrics=[],
                               average_scoring=["micro",
                                                "macro",
                                                "weighted"],
                               thresholds=None):
        """
        X/y:
            Feature matrix/Target data vector.

        title:
            Adds to the column 'Metric Score'.

        sub_dir:
            Specify a subdirectory to append to the output path of the file.

        file_name:
            Name of the file. If set to 'None' then don't create the file.

        custom_metrics:
            Pass the name of metric(s) and the function definition(s) in a
            dictionary.

        ignore_metrics:
            Specify the default metrics to not apply to the classification
            analysis.
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

        Returns/Desc:
            Creates/displays a dataframe object based on the model's
            predictions on the feature matrix compared to target data.
        """

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
        if len(custom_metrics.keys()):
            metric_functions.update(custom_metrics)

        # Evaluate model on metrics
        evaluation_report = dict()
        for metric_name in metric_functions:
            for average_score in average_scoring:

                model_predictions = self.get_model_prediction(X,
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

                except ValueError:
                    pass

        warnings.filterwarnings('default')

        if len(title) > 0:
            index_name = f"Metric Scores ({title})"
        else:
            index_name = "Metric Scores"

        # ---
        evaluation_report = pd.DataFrame({index_name:
                                              [f'{metric_score:.4f}'
                                               for metric_score
                                               in evaluation_report.values()]},
                                         index=list(evaluation_report.keys()))

        if self.__notebook_mode:
            display(evaluation_report)
        else:
            print(evaluation_report)

        # Create image file
        if file_name:
            df_to_image(evaluation_report,
                        self.get_output_folder(),
                        sub_dir,
                        convert_to_file_name(file_name),
                        col_width=20,
                        show_index=True,
                        format_float_pos=4)


    def classification_error_analysis(self,
                                      X,
                                      y,
                                      sub_dir="",
                                      thresholds=None,
                                      display_graphs=False):
        model_predictions = self.get_model_prediction(X,
                                                      thresholds=thresholds)

        if display_graphs:
            print("\n\n" + "*" * 10 +
                  "Correctly predicted analysis"
                  + "*" * 10 + "\n")
        else:
            print("\n\n" + "*" * 10 +
                  "Generating graphs for model's correctly predicted..." +
                  "*" * 10 + "\n")
        DataAnalysis(pd.DataFrame(X[model_predictions == y],
                     columns=self.__df_features.get_all_features()),
                     self.__df_features,
                     overwrite_full_path=self.get_output_folder() +
                                         sub_dir+ "/Correctly Predicted Data/",
                     missing_data_visuals=False,
                     notebook_mode=display_graphs)


        if display_graphs:
            print("\n\n" + "*" * 10 +
                  "Incorrectly predicted analysis"
                  + "*" * 10 + "\n")
        else:
            print("\n\n" + "*" * 10 +
                  "Generating graphs for model's incorrectly predicted..." +
                  "*" * 10 + "\n")

        DataAnalysis(pd.DataFrame(X[model_predictions != y],
                     columns=self.__df_features.get_all_features()),
                     self.__df_features,
                     overwrite_full_path=self.get_output_folder() +
                                         sub_dir + "/Incorrectly Predicted Data/",
                     missing_data_visuals=False,
                     notebook_mode=display_graphs)



