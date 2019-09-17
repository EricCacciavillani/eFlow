from eflow._hidden.Objects.enum import enum
from eflow.utils.SysUtils import create_plt_png, convert_to_filename, \
    df_to_image, write_object_text_to_file, get_unique_directory_path, \
    pickle_object_to_file
from eflow._hidden.Objects.FileOutput import *
from eflow._hidden.CustomExc import *
from eflow.Analysis.DataAnalysis import *

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
import pickle
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt


class ClassificationAnalysis(FileOutput):

    def __init__(self,
                 model,
                 model_name,
                 pred_funcs_dict,
                 project_name="Classification analysis_objects",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 target_classes=None,
                 df_features=None,
                 columns=[],
                 save_model=True):
        """
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

        project_name:
            Creates a parent or "project" folder in which all sub-directories
            will be inner nested.

        overwrite_full_path:
            Overwrites the path to the parent folder.

        notebook_mode:
            If in a python notebook display in the notebook.

        target_classes:
            Specfied list/np.array of targeted classes the model predicts. If set to
            none then it will attempt to pull from the sklearn's default attribute
            '.classes_'.

        df_features:
            DataFrameTypeHolder object. If initalized we can run correct/error
            analysis on the dataframe. Will save object in a pickle file and provided columns
            if initalized and df_features is not initalized.

        columns:
            Will overwrite over df_features (DataFrameTypeHolder) regardless of wether

        Returns/Desc:
            Evaluates the given model based on the prediction functions pased to it.
            Saves the model and other various graphs/dataframes for evaluation.
        """

        # Init any parent objects
        FileOutput.__init__(self,
                            f'{project_name}/{model_name}',
                            overwrite_full_path)

        # Init objects without pass by refrence
        self.__model = copy.deepcopy(model)
        self.__model_name = copy.deepcopy(model_name)
        self.__notebook_mode = copy.deepcopy(notebook_mode)
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
                                  self.get_output_folder(),
                                  f'{self.__model_name}')

        # ---
        check_create_dir_structure(self.get_output_folder(),
                                   "Extras")
        # Save predicted classes
        write_object_text_to_file(self.__target_values,
                                  self.get_output_folder() + "Extras",
                                  "_Classes")

        # Save features and or df_features object
        if columns or df_features:
            if columns:
                write_object_text_to_file(columns,
                                          self.get_output_folder() + "Extras",
                                          "_Features")
            else:
                write_object_text_to_file(df_features.get_all_features(),
                                          self.get_output_folder() + "Extras",
                                          "_Features")
                pickle_object_to_file(self.__model,
                                      self.get_output_folder() + "Extras",
                                      "_df_features")

        # Find the 'type' of each prediction. Probabilities or Predictions
        if self.__pred_funcs_dict:
            for pred_name, pred_func in self.__pred_funcs_dict.items():
                model_output = pred_func(
                    np.reshape(X_train[0],
                               (-1, X_train.shape[1])))[0]

                # Confidence / Probability (Continuous output)
                if isinstance(model_output, list) or isinstance(model_output,
                                                                np.ndarray):
                    self.__pred_funcs_types[pred_name] = "Probabilities"

                    # Classification (Discrete output)
                else:
                    self.__pred_funcs_types[pred_name] = "Predictions"
        else:
            raise RequiresPredictionMethods

    def __get_model_prediction(self,
                               pred_name,
                               X,
                               thresholds=None):
        """
        X:
            Feature matrix.

        pred_name:
            The name of the prediction function in questioned stored in 'self.__pred_funcs_dict'

        thresholds:
            If the model outputs a probability list/numpy array then we apply
            thresholds to the ouput of the model.

        Returns/Desc:
            Returns back a predicted value based for a given matrix.
            Handles prediction function 'types' Predictions and Probabilities.
            Helps streamline the entire process of evaluating classes.
        """

        # Must be a prediction function
        if self.__pred_funcs_types[pred_name] == "Predictions":
            return self.__pred_funcs_dict[pred_name](X)

        elif self.__pred_funcs_types[pred_name] == "Probabilities":
            model_output = self.__get_model_probas(pred_name,
                                                   X,
                                                   thresholds)

            return np.asarray([self.__target_values[np.argmax(proba)]
                               for proba in model_output])
        else:
            raise UnknownModelOutputType

    def __get_model_probas(self,
                           pred_name,
                           X,
                           thresholds=None):
        """
        X:
            Feature matrix.

        pred_name:
            The name of the prediction function in questioned stored in 'self.__pred_funcs_dict'

        thresholds:
            If the model outputs a probability list/numpy array then we apply
            thresholds to the ouput of the model.

        Returns/Desc:
            Returns back a series of values between 0-1 to represent it's confidence.
            Invokes an error if the prediction function call is anything but a Probabilities
            call.
        """

        if self.__pred_funcs_types[pred_name] == "Probabilities":
            model_output = self.__pred_funcs_dict[pred_name](X)

            # Validate probabilities
            if thresholds:
                if isinstance(thresholds, list) or \
                        isinstance(thresholds, np.ndarray):
                    if sum(thresholds) != 1:
                        print("Thresholds didn't add up to 100%! "
                              "This may cause issues in your results!")
                    if len(thresholds) != len(self.__target_values):
                        raise ThresholdLength
                else:
                    raise ThresholdType

            # ---
            if isinstance(model_output, list):
                model_output = np.asarray(model_output)

            if isinstance(model_output, np.ndarray):
                if thresholds:
                    model_output = model_output - np.asarray(thresholds)

            return model_output
        else:
            raise ProbasNotPossible

    def __create_sub_dir_with_thresholds(self,
                                         pred_name,
                                         dataset_name,
                                         thresholds):
        """
        pred_name:
            The prediction function's name.

        dataset_name:
            The passed in dataset's name.

        thresholds:
            If the model outputs a probability list/numpy array then we apply
            thresholds to the ouput of the model.

        Returns/Desc:
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
                    threshold_dir = self.get_output_folder()
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
                                                 directory_pth,
                                                 thresholds):
        """
        directory_pth:
            Path to the given folder where the "_Thresholds.txt"

        thresholds:
            If the model outputs a probability list/numpy array then we apply
            thresholds to the ouput of the model.

        Returns/Desc:
            Compare the thresholds object to the text file; returns true if
            the file exists and the object's value matches up.
        """

        file_directory = correct_directory_path(directory_pth)

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

    def peform_analysis(self,
                        X,
                        y,
                        dataset_name,
                        thresholds_matrix=None,
                        normalize_confusion_matrix=True,
                        ignore_metrics=[],
                        custom_metrics=dict(),
                        average_scoring=["micro",
                                         "macro",
                                         "weighted"],
                        display_analysis_graphs=False):
        """
        X/y:
            Feature matrix/Target data vector.

        dataset_name:
            The dataset's name.

        thresholds_matrix:
            Lists of list objects containing thresholds to iterate through
            for a model functon that is of type "Probabilities".

        normalize_confusion_matrix:
            Normalize the confusion matrix buckets.

        ignore_metrics:
            Specify set metrics to ignore. (F1-Score, Accuracy etc).

        ignore_metrics:
            Specify the default metrics to not apply to the classification
            analysis.
                * Precision
                * MCC
                * Recall
                * F1-Score
                * Accuracy

        custom_metrics:
            Pass the name of metric(s) and the function definition(s) in a
            dictionary.

        average_scoring:
            Determines the type of averaging performed on the data.

        display_analysis_graphs:
            Controls visual display of error error analysis if it is able to run.

        Returns/Desc:
            Performs all classification functionality with the provided feature
            data and target data.
                * plot_precision_recall_curve
                * classification_evaluation
                * plot_confusion_matrix
        """
        if isinstance(thresholds_matrix, np.ndarray):
            thresholds_matrix = thresholds_matrix.tolist()

        if not thresholds_matrix:
            thresholds_matrix = list()

        if isinstance(thresholds_matrix, list) and not isinstance(
                thresholds_matrix[0], list):
            thresholds_matrix = list(thresholds_matrix)

        thresholds_matrix.append(None)

        print("\n\n" + "---" * 10 + f'{dataset_name}' + "---" * 10)

        for pred_name, pred_type in self.__pred_funcs_types.items():
            print(f"Now running classification on {pred_name}")

            for thresholds in thresholds_matrix:
                if pred_type == "Predictions":
                    thresholds = None

                self.classification_metrics(X,
                                            y,
                                            pred_name=pred_name,
                                            dataset_name=dataset_name,
                                            thresholds=thresholds,
                                            ignore_metrics=ignore_metrics,
                                            custom_metrics=custom_metrics,
                                            average_scoring=average_scoring)

                self.plot_confusion_matrix(X,
                                           y,
                                           pred_name=pred_name,
                                           dataset_name=dataset_name,
                                           title=tmp_filename,
                                           thresholds=thresholds,
                                           normalize=normalize_confusion_matrix)

                self.plot_classification_error_analysis(X,
                                                        y,
                                                        pred_name=pred_name,
                                                        dataset_name=dataset_name,
                                                        thresholds=thresholds)

                if pred_type == "Probabilities":
                    self.plot_precision_recall_curve(X,
                                                     y,
                                                     sub_dir=sub_dir,
                                                     title=tmp_filename,
                                                     filename=tmp_filename,
                                                     thresholds=thresholds)
                if self.__df_features:
                    self.classification_error_analysis(X,
                                                       y,
                                                       pred_name=pred_name,
                                                       dataset_name=dataset_name,
                                                       thresholds=thresholds,
                                                       display_graphs=display_graphs)

                    if pred_type == "Predictions":
                        break

    #     def plot_precision_recall_curve(self,
    #                                     X,
    #                                     y,
    #                                     pred_name,
    #                                     dataset_name,
    #                                     thresholds=None,
    #                                     sub_dir="",
    #                                     figsize=(10, 8),
    #                                     title=None,
    #                                     filename=None):
    #         """
    #         X/y:
    #             Feature matrix/Target data vector.

    #         sub_dir:
    #             Specify a subdirectory to append to the output path of the file.

    #         figsize:
    #             Plot's size.

    #         title:
    #             Title of the plot.

    #         filename:
    #             Name of the file. If set to 'None' then don't create the file.

    #         Returns/Desc:

    #         """
    #         if title:
    #             skplt.metrics.plot_precision_recall(self.get_model_probas(X),
    #                                                 y,
    #                                                 figsize=figsize,
    #                                                 title=title)
    #         else:
    #             skplt.metrics.plot_precision_recall(self.get_model_probas(X),
    #                                                 y,
    #                                                 figsize=figsize)

    #         if filename:
    #             create_plt_png(self.get_output_folder(),
    #                            sub_dir,
    #                            convert_to_filename(filename))

    #         if self.__notebook_mode:
    #             plt.show()
    #         plt.close()

    def plot_confusion_matrix(self,
                              X,
                              y,
                              pred_name,
                              dataset_name,
                              thresholds=None,
                              figsize=(10, 8),
                              title=None,
                              normalize=True):
        """
        X/y:
            Feature matrix/Target data vector.

        pred_name:

        dataset_name:
            The dataset's name.
        thresholds:

        figsize:

        title:

        normalize:

        Returns/Descr:

        """

        filename = f'Confusion Matrix: {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        warnings.filterwarnings('ignore')
        if title:
            skplt.metrics.plot_confusion_matrix(
                self.__get_model_prediction(pred_name,
                                            X,
                                            thresholds),
                y,
                figsize=figsize,
                title=title,
                normalize=normalize, )
        else:
            skplt.metrics.plot_confusion_matrix(
                self.__get_model_prediction(pred_name,
                                            X,
                                            thresholds),
                y,
                figsize=figsize,
                normalize=normalize)
        warnings.filterwarnings('default')

        create_plt_png(self.get_output_folder(),
                       sub_dir,
                       convert_to_filename(filename))

        if self.__notebook_mode:
            plt.show()
        plt.close()

    def classification_metrics(self,
                               X,
                               y,
                               pred_name,
                               dataset_name,
                               thresholds=None,
                               title="",
                               sub_dir="",
                               filename=None,
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
        if len(custom_metrics.keys()):
            metric_functions.update(custom_metrics)

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
        df_to_image(evaluation_report,
                    self.get_output_folder(),
                    sub_dir,
                    convert_to_filename(filename),
                    col_width=20,
                    show_index=True,
                    format_float_pos=4)

    def plot_classification_error_analysis(self,
                                           X,
                                           y,
                                           pred_name,
                                           dataset_name,
                                           thresholds=None,
                                           display_graphs=False):

        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        model_predictions = self.__get_model_prediction(pred_name,
                                                        X,
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
                                         sub_dir + "/Correctly Predicted Data/",
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

    def classification_report(self,
                              X,
                              y,
                              pred_name,
                              dataset_name,
                              thresholds=None):
        """
        X/y:

        pred_name:

        dataset_name:

        thresholds:
        """
        filename = f'Classification Report {dataset_name} on {self.__model_name}'
        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        report_df = pd.DataFrame(classification_report(y,
                                                       self.__get_model_prediction(
                                                           pred_name,
                                                           X,
                                                           thresholds),
                                                       output_dict=True))

        if self.__notebook_mode:
            display(report_df)
        else:
            print(report_df)

        df_to_image(report_df,
                    self.get_output_folder(),
                    sub_dir,
                    filename,
                    col_width=20,
                    show_index=True,
                    format_float_pos=4)