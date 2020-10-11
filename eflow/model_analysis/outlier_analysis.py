from eflow.utils.sys_utils import *
from eflow._hidden.parent_objects import ModelAnalysis
from eflow.utils.pandas_utils import zcore_remove_outliers

import pandas as pd
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import numpy as np

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class OutlierAnalysis(ModelAnalysis):

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
                 df_features,
                 project_sub_dir="Outlier Analysis",
                 overwrite_full_path=None,
                 save_models=True,
                 notebook_mode=False):

        # Init parent object
        ModelAnalysis.__init__(self,
                               f'{dataset_name}/{project_sub_dir}/{model_name}',
                               overwrite_full_path)

        self.__feature_order = copy.deepcopy(feature_order)

        self.__model = copy.deepcopy(model)
        self.__model_name = copy.deepcopy(model_name)

        self.__df_features = copy.deepcopy(df_features)
        self.__notebook_mode = copy.deepcopy(notebook_mode)

        # Determines if the perform was called
        self.__called_from_perform = False

        del model

        # Attempt to save machine learning model
        if save_models:
            try:
                pickle_object_to_file(model,
                                      self.folder_path,
                                      model_name)
            except:
                pass

        create_dir_structure(self.folder_path,
                             "_Extras")

    def model_decision_outliers(self):

        return None


    def graph_decision_outliers(self,
                                X,
                                heavy_outlier_zscore=float("inf"),
                                medium_outlier_zscore=float("inf"),
                                save_file=True):

        model_decisions = self.__model.decision_function(X)

        zscore_val = heavy_outlier_zscore
        zscore_series = pd.Series((model_decisions - model_decisions.mean()) /
                                  model_decisions.std(ddof=0))

        bool_series = zscore_series.between(zscore_val * -1, zscore_val)
        hv_outlier_index_list = bool_series[
            bool_series == False].index.tolist()

        hv_outlier_dict = dict(zip(hv_outlier_index_list,list(zscore_series)))
        tmp_df = pd.DataFrame.from_dict(hv_outlier_dict,
                                        orient='index',columns=["Z-Scores"])

        tmp_df.sort_values(by=['Z-Scores'], inplace=True)
        tmp_df.to_csv(self.folder_path +  f"_Extras/Heavy Outliers and Inlier with a abs zscore {zscore_val}.csv")


        # SAVE HERE

        del zscore_series, hv_outlier_dict, tmp_df

        zscore_val = medium_outlier_zscore
        zscore_series = pd.Series((model_decisions[bool_series] -
                                   model_decisions[bool_series].mean()) /
                                  model_decisions[bool_series].std(ddof=0))

        bool_series = zscore_series.between(zscore_val * -1,
                                            zscore_val)
        md_outlier_index_list = bool_series[
            bool_series == False].index.tolist()


        md_outlier_dict = dict(zip(md_outlier_index_list, list(zscore_series)))

        tmp_df = pd.DataFrame.from_dict(md_outlier_dict,
                                        orient='index', columns=["Z-Scores"])

        tmp_df.sort_values(by=['Z-Scores'], inplace=True)
        tmp_df.to_csv(
            self.folder_path + f"_Extras/Medium Outliers and Inlier with a abs zscore of {zscore_val}.csv")

        # SAVE HERE

        del zscore_series, md_outlier_dict, tmp_df,bool_series

        for shading_pos_neg in [True,False]:
            self.__helper_decision_outlier_graph(model_decisions,
                                                 title=f"{self.__model_name} decision function",
                                                 filename=f"{self.__model_name} decision function",
                                                 shading_pos_neg=shading_pos_neg,
                                                 save_file=save_file)

            title_and_filename = f"{self.__model_name} decision function with heavy outlier removed (Zscore of {heavy_outlier_zscore})"

            if not shading_pos_neg:
                title_and_filename += " show zscore boundaries"

            self.__helper_decision_outlier_graph(model_decisions,
                                                 hv_outlier_index_list=hv_outlier_index_list,
                                                 title=title_and_filename,
                                                 filename=title_and_filename,
                                                 shading_pos_neg=shading_pos_neg,
                                                 save_file=save_file)

            title_and_filename = f"{self.__model_name} decision function with medium (Zscore of {medium_outlier_zscore}) and heavy outlier removed (Zscore of {heavy_outlier_zscore})"

            if not shading_pos_neg:
                title_and_filename += " show zscore boundaries"

            self.__helper_decision_outlier_graph(model_decisions,
                                                 hv_outlier_index_list=hv_outlier_index_list,
                                                 md_outlier_index_list=md_outlier_index_list,
                                                 title=title_and_filename,
                                                 filename=title_and_filename,
                                                 shading_pos_neg=shading_pos_neg,
                                                 save_file=save_file)




    def __helper_decision_outlier_graph(self,
                                        model_decisions,
                                        hv_outlier_index_list=[],
                                        md_outlier_index_list=[],
                                        title="default",
                                        filename="default",
                                        shading_pos_neg=True,
                                        save_file=True):

        outlier_val = min(np.delete(model_decisions,
                                    md_outlier_index_list + hv_outlier_index_list))
        inlier_val = max(np.delete(model_decisions,
                                   md_outlier_index_list + hv_outlier_index_list))

        model_decisions = np.delete(model_decisions,
                                    md_outlier_index_list + hv_outlier_index_list)

        del hv_outlier_index_list, md_outlier_index_list


        plt.figure(figsize=(10, 10))
        ax = sns.distplot(model_decisions, kde=True,
                          hist_kws={'edgecolor': 'black', "rwidth": .9, },
                          bins=16)

        ymax = ax.get_ylim()[1] - (ax.get_ylim()[1] * .03)

        if shading_pos_neg:
            outlier_val = 0

            ax.annotate('Outlier boundaries', xy=(outlier_val, 0),
                        xytext=(outlier_val, ymax),
                        arrowprops=dict(facecolor='red', alpha=.5))

            rect = patches.Rectangle((outlier_val, 0), ax.get_xlim()[1],
                                     ymax - 0, facecolor='blue', alpha=0.07, )
            ax.add_patch(rect)

            rect = patches.Rectangle((ax.get_xlim()[0], 0),
                                     outlier_val - ax.get_xlim()[0], ymax - 0,
                                     facecolor='#b93c43', alpha=0.1)
            ax.add_patch(rect)

            plt.legend(['Inliers', f'Abs Z-Score higher than '],
                       bbox_to_anchor=(1.02, 1), loc='upper left')
        else:

            ax.annotate('Outlier boundaries', xy=(outlier_val, 0),
                        xytext=(outlier_val, ymax),
                        arrowprops=dict(facecolor='red', alpha=.5))
            ax.annotate('Inlier boundaries', xy=(inlier_val, 0),
                        xytext=(inlier_val, ymax),
                        arrowprops=dict(facecolor='red', alpha=.5))

            rect = patches.Rectangle((outlier_val, 0),
                                     inlier_val - outlier_val, ymax - 0,
                                     facecolor='blue', alpha=0.07, )
            ax.add_patch(rect)

            rect = patches.Rectangle((ax.get_xlim()[0], 0),
                                     outlier_val - ax.get_xlim()[0], ymax - 0,
                                     facecolor='#b93c43', alpha=0.1)
            ax.add_patch(rect)

            rect = patches.Rectangle((ax.get_xlim()[1], 0),
                                     inlier_val - ax.get_xlim()[1], ymax - 0,
                                     facecolor='#b93c43', alpha=0.1)
            ax.add_patch(rect)


        plt.legend(['Inliers', f'Abs Z-Score higher than '],
                   bbox_to_anchor=(1.02, 1), loc='upper left')
        leg = ax.get_legend()

        if len(leg.legendHandles) == 2:
            leg.legendHandles[0].set_color('blue')
            leg.legendHandles[1].set_color('#b93c43')

        for lh in leg.legendHandles:
            lh.set_alpha(.4)

        plt.title(title)

        if save_file:
            self.save_plot(filename=filename)

        plt.show()
        plt.close("all")
