from eFlow._Hidden.Objects.enum import enum
from eFlow.Utils.SysUtils import create_plt_png, convert_to_file_name, df_to_image
from eFlow._Hidden.Objects.FileOutput import *
from eFlow._Hidden.Constants import PREDICTION_TYPES

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
                 X_train=None,
                 y_train=None,
                 X_test=None,
                 y_test=None,
                 X_val=None,
                 y_val=None,
                 prediction_type=PREDICTION_TYPES.CLASSIFICATION,
                 project_name="Supervised Analysis",
                 overwrite_full_path=None,
                 notebook_mode=True):
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

        # Init objectss by pass by refrence
        self.__model = copy.deepcopy(model)
        self.__model_name = copy.deepcopy(model_name)
        self.__notebook_mode = copy.deepcopy(notebook_mode)
        self.__prediction_type = copy.deepcopy(prediction_type)
        self.__binary_classification = None

        # Classification model
        if prediction_type == PREDICTION_TYPES.CLASSIFICATION:

            self.__classified_target_values = set()
            for y in [y_train, y_test, y_val]:
                if y is not None:
                    self.__classified_target_values |= set(y)

            if len(self.__classified_target_values) == 2:
                self.__binary_classification = True
            else:
                self.__binary_classification = False


            if X_train is not None and y_train is not None:
                print("\n\n---" * 10 + "Training data" + "---" * 10)
                self.classification_analysis(X_train,
                                             y_train,
                                             dataset_name="Train data")

            if X_test is not None and y_test is not None:
                print("\n\n" + "---" * 10 + "Test data" + "---" * 10)
                self.classification_analysis(X_test,
                                             y_test,
                                             dataset_name="Test data")
            if X_val is not None and y_val is not None:
                print("\n\n" + "---" * 10 + "Validation data" + "---" * 10)
                self.classification_analysis(X_val,
                                             y_val,
                                             dataset_name="Validation data")


        # Regression model
        elif prediction_type == PREDICTION_TYPES.REGRESSION:
            pass
        else:
            print("ERROR")

    def classification_analysis(self,
                                X,
                                y,
                                dataset_name,
                                normalize_confusion_matrix=True,
                                ignore_metrics=[],
                                custom_metrics=dict(),
                                average_scoring=["micro",
                                                 "macro",
                                                 "weighted"]):
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

            tmp_file_name = f'Precision Recall Curve with Probabilities ' + \
                            f'on {dataset_name}'
            self.plot_precision_recall_curve(X,
                                             y,
                                             sub_dir=f'{dataset_name}',
                                             title=tmp_file_name,
                                             filename=tmp_file_name)

            self.classification_evaluation(X,
                                           y,
                                           title=dataset_name,
                                           sub_dir=f'{dataset_name}',
                                           ignore_metrics=ignore_metrics,
                                           custom_metrics=custom_metrics,
                                           file_name=f'Evaluation on ' + \
                                           f'{dataset_name}',
                                           average_scoring=average_scoring)

            self.plot_confusion_matrix(X,
                                       y,
                                       sub_dir=f'{dataset_name}',
                                       title=f'Confusion Matrix: ' + \
                                       f'{dataset_name}',
                                       normalize=normalize_confusion_matrix)

            y_probas = self.__model.predict_proba(X)

            if self.__binary_classification:
                skplt.metrics.plot_ks_statistic(y,
                                                y_probas,
                                                figsize=(10, 8))

        else:
            print(f'Model prediction is not a classification problem.' + \
                  f'It is a {self.__prediction_type}')


    def plot_precision_recall_curve(self,
                                    X,
                                    y,
                                    sub_dir="",
                                    figsize=(10, 8),
                                    title=None,
                                    filename=None):
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
            skplt.metrics.plot_precision_recall(y,
                                                self.__model.predict_proba(X),
                                                figsize=figsize,
                                                title=title)
        else:
            skplt.metrics.plot_precision_recall(self.__model.predict_proba(X),
                                                y,
                                                figsize=figsize)

        if filename:
            create_plt_png(FileOutput.get_output_folder(self),
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
                              normalize=True):
        warnings.filterwarnings('ignore')
        if title:
            skplt.metrics.plot_confusion_matrix(self.__model.predict(X),
                                                y,
                                                figsize=figsize,
                                                title=title,
                                                normalize=normalize)
        else:
            skplt.metrics.plot_confusion_matrix(self.__model.predict(X),
                                                y,
                                                figsize=figsize,
                                                normalize=normalize)
        warnings.filterwarnings('default')

        if file_name:
            create_plt_png(FileOutput.get_output_folder(self),
                           sub_dir,
                           convert_to_file_name(file_name))

        if self.__notebook_mode:
            plt.show()
        plt.close()

    def classification_evaluation(self,
                                  X,
                                  y,
                                  title="",
                                  sub_dir="",
                                  file_name=None,
                                  custom_metrics=dict(),
                                  ignore_metrics=[],
                                  average_scoring=["micro",
                                                   "macro",
                                                   "weighted"]):
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

                model_predictions = self.__model.predict(X)
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
                        FileOutput.get_output_folder(self),
                        sub_dir,
                        convert_to_file_name(file_name),
                        show_index=True,
                        format_float_pos=4)


