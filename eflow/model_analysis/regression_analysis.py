from eflow.utils.sys_utils import *
from eflow.utils.pandas_utils import df_to_image
from eflow.utils.image_processing_utils import create_plt_png
from eflow._hidden.parent_objects import ModelAnalysis
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

class RegressionAnalysis(ModelAnalysis):

    """
        Analyzes a classification model's result's based on the prediction
        function(s) passed to it. Creates graphs and tables to be saved in directory
        structure.
    """

    def __init__(self,
                 dataset_name,
                 model,
                 model_name,
                 feature_order,
                 target_feature,
                 df_features,
                 project_sub_dir="Regression Analysis",
                 overwrite_full_path=None,
                 target_classes=None,
                 save_model=True,
                 notebook_mode=False):
        """
        Args:
            dataset_name:
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            model:
                A fitted supervised machine learning model.

            model_name:
                The name of the model in string form.

            feature_order: collection object
                Features names in proper order to re-create the pandas dataframe.

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
        ModelAnalysis.__init__(self,
                               f'{dataset_name}/{project_sub_dir}/Target Feature: {target_feature}/{model_name}',
                               overwrite_full_path)

        # Init objects without pass by refrence

        # Remove target feature from feature order when trying to recreate dataframe
        self.__target_feature = copy.deepcopy(target_feature)
        self.__feature_order = copy.deepcopy(feature_order)

        if self.__target_feature in self.__feature_order:
            self.__feature_order.remove(self.__target_feature)

        self.__model = copy.deepcopy(model)

        self.__model_name = copy.deepcopy(model_name)
        self.__target_values = copy.deepcopy(target_classes)
        self.__df_features = copy.deepcopy(df_features)
        self.__pred_funcs_types = dict()
        self.__notebook_mode = copy.deepcopy(notebook_mode)

        # Determines if the perform was called
        self.__called_from_perform = False

        # Attempt to save machine learning model
        try:
            if save_model:
                pickle_object_to_file(self.__model,
                                      self.folder_path,
                                      f'{self.__model_name}')
        except:
            pass
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

    def perform_analysis(self,
                         X,
                         y,
                         dataset_name,
                         regression_error_analysis=False,
                         regression_correct_analysis=False,
                         ignore_metrics=[],
                         custom_metrics_dict=dict(),
                         display_visuals=True):
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

            classification_error_analysis: bool
                Perform feature analysis on data that was incorrectly predicted.

            classification_correct_analysis: bool
                Perform feature analysis on data that was correctly predicted.

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
        try:
            self.__called_from_perform = True

            self.generate_matrix_meta_data(X,
                                           dataset_name + "/_Extras")

            print("\n\n" + "---" * 10 + f'{dataset_name}' + "---" * 10)
        finally:
            self.__called_from_perform = False

    def regression_metrics(self,
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
            if self.__notebook_mode:
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

            if not self.__called_from_perform:
                self.generate_matrix_meta_data(X,
                                               dataset_name + "/_Extras")


    def regression_correct_analysis(self,
                                    X,
                                    y,
                                    pred_name,
                                    dataset_name,
                                    thresholds=None,
                                    display_visuals=True,
                                    save_file=True,
                                    aggerate_target=False,
                                    display_print=True,
                                    suppress_runtime_errors=True,
                                    aggregate_target_feature=True,
                                    selected_features=None,
                                    extra_tables=True,
                                    statistical_analysis_on_aggregates=True):
        """
        Desc:
            Compares the actual target value to the predicted value and performs
            analysis of all the data.

        Args:
            X: np.matrix or lists of lists
                Feature matrix.

            y: collection object
                Target data vector.

            pred_name: str
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name: str
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            feature_order: collection object
                Features names in proper order to re-create the pandas dataframe.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            extra_tables: bool
                When handling two types of features if set to true this will
                    generate any extra tables that might be helpful.
                    Note -
                        These graphics may create duplicates if you already applied
                        an aggregation in 'perform_analysis'

            aggregate_target_feature: bool
                Aggregate the data of the target feature if the data is
                non-continuous data.

                Note
                    In the future I will have this also working with continuous
                    data.

            selected_features: collection object of features
                Will only focus on these selected feature's and will ignore
                the other given features.

            statistical_analysis_on_aggregates: bool
                If set to true then the function 'statistical_analysis_on_aggregates'
                will run; which aggregates the data of the target feature either
                by discrete values or by binning/labeling continuous data.
        """

        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        model_predictions = self.__get_model_prediction(pred_name,
                                                        X,
                                                        thresholds=thresholds)

        if sum(model_predictions != y) == len(y):
            print("Your model predicted everything correctly for this dataset! No correct analysis needed!")
            print("Also sorry for your model...zero correct? Dam...")
        else:
            print("\n\n" + "*" * 10 +
                  "Generating graphs for when the model predicted correctly" +
                  "*" * 10 + "\n")

            # Generate error dataframe
            correct_df = pd.DataFrame.from_records(X[model_predictions == y])
            correct_df.columns = self.__feature_order
            correct_df[self.__target_feature] = y[model_predictions == y]

            # Directory path
            create_dir_structure(self.folder_path,
                                 sub_dir + "/Correctly Predicted Data/All Correct Data")
            output_path = f"{self.folder_path}/{sub_dir}/Correctly Predicted Data"

            # Create feature analysis
            feature_analysis = FeatureAnalysis(self.__df_features,
                                               overwrite_full_path=output_path + "/All Correct Data")
            feature_analysis.perform_analysis(correct_df,
                                              dataset_name=dataset_name,
                                              target_features=[self.__target_feature],
                                              save_file=save_file,
                                              selected_features=selected_features,
                                              suppress_runtime_errors=suppress_runtime_errors,
                                              display_print=display_print,
                                              display_visuals=display_visuals,
                                              dataframe_snapshot=False,
                                              aggregate_target_feature=aggregate_target_feature,
                                              extra_tables=extra_tables,
                                              statistical_analysis_on_aggregates=statistical_analysis_on_aggregates)

            # Aggregate target by predicted and actual
            if aggerate_target:
                targets = set(y)
                for pred_target in targets:
                    if pred_target != pred_target:
                        create_dir_structure(output_path,
                                             f"/Actual and Predicted:{pred_target}")

                        # Create predicted vs actual dataframe
                        tmp_df = copy.deepcopy(correct_df[correct_df[
                                                              self.__target_feature] == pred_target])

                        if tmp_df.shape[0]:
                            # Create feature analysis directory structure with given graphics
                            feature_analysis = FeatureAnalysis(
                                self.__df_features,
                                overwrite_full_path=f"/Actual and Predicted:{pred_target}")
                            feature_analysis.perform_analysis(tmp_df,
                                                              dataset_name=dataset_name,
                                                              target_features=[
                                                                  self.__target_feature],
                                                              save_file=save_file,
                                                              selected_features=selected_features,
                                                              suppress_runtime_errors=suppress_runtime_errors,
                                                              display_print=display_print,
                                                              display_visuals=display_visuals,
                                                              dataframe_snapshot=False,
                                                              extra_tables=extra_tables,
                                                              statistical_analysis_on_aggregates=statistical_analysis_on_aggregates)

    def regression_error_analysis(self,
                                  X,
                                  y,
                                  pred_name,
                                  dataset_name,
                                  thresholds=None,
                                  display_visuals=True,
                                  save_file=True,
                                  aggerate_target=False,
                                  display_print=True,
                                  suppress_runtime_errors=True,
                                  aggregate_target_feature=True,
                                  selected_features=None,
                                  extra_tables=True,
                                  statistical_analysis_on_aggregates=True):
        """
        Desc:
            Compares the actual target value to the predicted value and performs
            analysis of all the data.

        Args:
            X: np.matrix or lists of lists
                Feature matrix.

            y: collection object
                Target data vector.

            pred_name: str
                The name of the prediction function in questioned
                stored in 'self.__pred_funcs_dict'

            dataset_name: str
                The dataset's name; this will create a sub-directory in which your
                generated graph will be inner-nested in.

            feature_order: collection object
                Features names in proper order to re-create the pandas dataframe.

            thresholds:
                If the model outputs a probability list/numpy array then we apply
                thresholds to the ouput of the model.
                For classification only; will not affect the direct output of
                the probabilities.

            display_visuals: bool
                Boolean value to whether or not to display visualizations.

            display_print: bool
                Determines whether or not to print function's embedded print
                statements.

            save_file: bool
                Boolean value to whether or not to save the file.

            dataframe_snapshot: bool
                Boolean value to determine whether or not generate and compare a
                snapshot of the dataframe in the dataset's directory structure.
                Helps ensure that data generated in that directory is correctly
                associated to a dataframe.

            suppress_runtime_errors: bool
                If set to true; when generating any graphs will suppress any runtime
                errors so the program can keep running.

            extra_tables: bool
                When handling two types of features if set to true this will
                    generate any extra tables that might be helpful.
                    Note -
                        These graphics may create duplicates if you already applied
                        an aggregation in 'perform_analysis'

            aggregate_target_feature: bool
                Aggregate the data of the target feature if the data is
                non-continuous data.

                Note
                    In the future I will have this also working with continuous
                    data.

            selected_features: collection object of features
                Will only focus on these selected feature's and will ignore
                the other given features.

            statistical_analysis_on_aggregates: bool
                If set to true then the function 'statistical_analysis_on_aggregates'
                will run; which aggregates the data of the target feature either
                by discrete values or by binning/labeling continuous data.
        """

        sub_dir = self.__create_sub_dir_with_thresholds(pred_name,
                                                        dataset_name,
                                                        thresholds)

        model_predictions = self.__get_model_prediction(pred_name,
                                                        X,
                                                        thresholds=thresholds)

        if sum(model_predictions == y) == len(y):
            print("Your model predicted everything correctly for this dataset! No error analysis needed!")
        else:
            print("\n\n" + "*" * 10 +
                  "Generating graphs for when the model predicted incorrectly" +
                  "*" * 10 + "\n")

            # Generate error dataframe
            error_df = pd.DataFrame.from_records(X[model_predictions != y])
            error_df.columns = self.__feature_order
            error_df[self.__target_feature] = y[model_predictions != y]

            # Directory path
            create_dir_structure(self.folder_path,
                                 sub_dir + "/Incorrectly Predicted Data/All Incorrect Data")
            output_path = f"{self.folder_path}/{sub_dir}/Incorrectly Predicted Data"

            # Create feature analysis
            feature_analysis = FeatureAnalysis(self.__df_features,
                                               overwrite_full_path=output_path + "/All Incorrect Data")
            feature_analysis.perform_analysis(error_df,
                                              dataset_name=dataset_name,
                                              target_features=[self.__target_feature],
                                              save_file=save_file,
                                              selected_features=selected_features,
                                              suppress_runtime_errors=suppress_runtime_errors,
                                              display_print=display_print,
                                              display_visuals=display_visuals,
                                              dataframe_snapshot=False,
                                              aggregate_target_feature=aggregate_target_feature,
                                              extra_tables=extra_tables,
                                              statistical_analysis_on_aggregates=statistical_analysis_on_aggregates)

            # Aggregate target by predicted and actual
            if aggerate_target:
                targets = set(y)
                prediction_feature = self.__target_feature + "_MODEL_PREDICTIONS_"
                error_df[prediction_feature] = model_predictions[model_predictions != y]
                for actual_target in targets:
                    for pred_target in targets:
                        if pred_target != actual_target:
                            create_dir_structure(output_path,
                                                 f"/Predicted:{pred_target} Actual: {actual_target}")

                            # Create predicted vs actual dataframe
                            tmp_df = copy.deepcopy(error_df[error_df[
                                                                self.__target_feature] == actual_target][
                                                       error_df[
                                                           prediction_feature] == pred_target])

                            tmp_df.drop(columns=[prediction_feature],
                                        inplace=True)
                            if tmp_df.shape[0]:
                                # Create feature analysis directory structure with given graphics
                                feature_analysis = FeatureAnalysis(
                                    self.__df_features,
                                    overwrite_full_path=f"{output_path}/Predicted:{pred_target} Actual: {actual_target}")
                                feature_analysis.perform_analysis(tmp_df,
                                                                  dataset_name=dataset_name,
                                                                  target_features=[
                                                                      self.__target_feature],
                                                                  save_file=save_file,
                                                                  selected_features=selected_features,
                                                                  suppress_runtime_errors=suppress_runtime_errors,
                                                                  display_print=display_print,
                                                                  display_visuals=display_visuals,
                                                                  dataframe_snapshot=False,
                                                                  extra_tables=extra_tables,
                                                                  statistical_analysis_on_aggregates=statistical_analysis_on_aggregates)