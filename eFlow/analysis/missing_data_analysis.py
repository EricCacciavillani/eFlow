from eflow._hidden.Objects.FileOutput import *
import copy
from IPython.display import display
import pandas as pd
import seaborn as sns
import missingno as msno
from matplotlib import pyplot as plt
import warnings
from eflow.utils.sys_utils import df_to_image
from eflow._hidden.custom_warnings import DataFrameWarning
from eflow._hidden.Constants import GRAPH_DEFAULTS
from eflow.utils.sys_utils import create_plt_png, convert_to_filename

class MissingDataAnalysis(FileOutput):

    def __init__(self,
                 sub_dir="",
                 project_name="Missing Data",
                 overwrite_full_path=None,
                 notebook_mode=True):

        FileOutput.__init__(self,
                            f'{sub_dir}/{project_name}',
                            overwrite_full_path)
        self.__notebook_mode = copy.deepcopy(notebook_mode)
        self.__called_from_peform = False

    def __check_dataframe(self,
                          df):

        passed_check = True

        if not df.isnull().values.any():

            warnings.warn('The given object requires null data to visualize',
                          DataFrameWarning,
                          stacklevel=1000)
            passed_check = False

        if df.shape[0] == 0:
            warnings.warn('The given object requires null data to visualize',
                          DataFrameWarning,
                          stacklevel=1000)
            passed_check = False

        if not passed_check:
            print(
                "All functionality belonging to this object requies null data!")

        return passed_check

    def peform_analysis(self,
                        df,
                        dataset_name,
                        df_features=None,
                        display_visuals=None):

        self.__called_from_peform = False

        if df is not None:

            # All functionality is meaningless without getting past the
            # following check; exit function
            if not self.__check_dataframe(df):
                return
            else:
                self.__called_from_peform = True

            self.data_types_table(df,
                                  dataset_name,
                                  display_visuals=display_visuals)
            print("\n\n")

            self.missing_values_table(df,
                                      dataset_name,
                                      display_visuals=display_visuals)
            print("\n\n")

            self.plot_null_bar_graph(df,
                                     dataset_name,
                                     display_visuals=display_visuals)
            print("\n\n")

            self.plot_null_matrix_graph(df,
                                        dataset_name,
                                        display_visuals=display_visuals)
            print("\n\n")

            self.plot_null_heatmap_graph(df,
                                         dataset_name,
                                         display_visuals=display_visuals)
            print("\n\n")

            self.plot_null_dendrogram_graph(df,
                                            dataset_name,
                                            display_visuals=display_visuals)
            print("\n\n")

    def plot_null_matrix_graph(self,
                               df,
                               dataset_name,
                               save_file=True,
                               display_visuals=None,
                               filter=None,
                               n=0,
                               p=0,
                               sort=None,
                               figsize=(25, 10),
                               width_ratios=(15, 1),
                               color=(.027, .184, .373),
                               fontsize=16,
                               labels=None,
                               sparkline=True,
                               inline=False,
                               freq=None,
                               ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        save_file:
            Boolean value to wether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Descr (Taken from missingno):
            A matrix visualization of the nullity of the given DataFrame.
        """

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        if not display_visuals:
            display_visuals = self.__notebook_mode
            print("Generating graph for null matrix graph...")

        msno.matrix(df,
                    filter=filter,
                    n=n,
                    p=p,
                    sort=sort,
                    figsize=figsize,
                    width_ratios=width_ratios,
                    color=color,
                    fontsize=fontsize,
                    labels=labels,
                    sparkline=sparkline,
                    inline=inline,
                    freq=freq,
                    ax=ax)
        if save_file:
            create_plt_png(self.get_output_folder(),
                           f"{dataset_name}/Graphics",
                           convert_to_filename("Missing data matrix graph"))

        if display_visuals:
            plt.show()
        plt.close()


    def plot_null_bar_graph(self,
                            df,
                            dataset_name,
                            save_file=True,
                            display_visuals=None,
                            figsize=(24, 10),
                            fontsize=16,
                            labels=None,
                            log=False,
                            color="#072F5F",
                            inline=False,
                            filter=False,
                            n=0,
                            p=0,
                            sort=None,
                            ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        save_file:
            Boolean value to wether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Descr (Taken from missingno):
            A bar graph visualization of the nullity of the given DataFrame.
        """

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        if not display_visuals:
            display_visuals = self.__notebook_mode
            print("Generating graph for null bar graph...")

        msno.bar(df,
                 figsize=figsize,
                 log=log,
                 fontsize=fontsize,
                 labels=labels,
                 color=color,
                 inline=inline,
                 filter=filter,
                 n=n,
                 p=p,
                 sort=sort,
                 ax=ax)
        if save_file:
            create_plt_png(self.get_output_folder(),
                           f"{dataset_name}/Graphics",
                           convert_to_filename("Missing data bar graph"))

        if display_visuals:
            plt.show()
        plt.close()

    def plot_null_heatmap_graph(self,
                                df,
                                dataset_name,
                                save_file=True,
                                display_visuals=None,
                                inline=False,
                                filter=None,
                                n=0,
                                p=0,
                                sort=None,
                                figsize=(20, 12),
                                fontsize=16,
                                labels=True,
                                cmap='RdBu',
                                vmin=-1,
                                vmax=1,
                                cbar=True,
                                ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        save_file:
            Boolean value to wether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Descr (Taken from missingno):
            Presents a `seaborn` heatmap visualization of nullity correlation
            in the given DataFrame.
        """

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        if not display_visuals:
            display_visuals = self.__notebook_mode
            print("Generating graph for null heatmap...")

        msno.heatmap(df,
                     inline=inline,
                     filter=filter,
                     n=n,
                     p=p,
                     sort=sort,
                     figsize=figsize,
                     fontsize=fontsize,
                     labels=labels,
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     cbar=cbar,
                     ax=ax
                     )
        if save_file:
            create_plt_png(self.get_output_folder(),
                           f"{dataset_name}/Graphics",
                           convert_to_filename("Missing data heatmap graph on")
                           )

        if display_visuals:
            plt.show()
        plt.close()


    def plot_null_dendrogram_graph(self,
                                  df,
                                  dataset_name,
                                  save_file=True,
                                  method='average',
                                  filter=None,
                                  n=0,
                                  p=0,
                                  orientation=None,
                                  figsize=None,
                                  fontsize=16,
                                  inline=False,
                                  ax=None):
        # All credit to the following author for making the 'missingno' package
        # https://github.com/ResidentMario/missingno
        """
        df:
            Pandas dataframe object

        save_file:
            Boolean value to wether or not to save the file.

        Please read the offical documentation at:
        Link: https://github.com/ResidentMario/missingno

        Note:
            Changed the default color of the bar graph because I thought it
            was ugly.

        Returns/Descr (Taken from missingno):
            Fits a `scipy` hierarchical clustering algorithm to the given
            DataFrame's variables and visualizes the results as
            a `scipy` dendrogram.
        """

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        if not self.__notebook_mode:
            print("Generating graph for null dendrogram...")

        msno.dendrogram(df,
                        method=method,
                        filter=filter,
                        n=n,
                        p=p,
                        orientation=orientation,
                        figsize=figsize,
                        fontsize=fontsize,
                        inline=inline,
                        ax=ax)
        if save_file:
            create_plt_png(self.get_output_folder(),
                           f"{dataset_name}/Graphics",
                           convert_to_filename(f"Missing data dendrogram "
                                               f"graph {method}"))

        if self.__notebook_mode:
            plt.show()
        plt.close()

    def data_types_table(self,
                         df,
                         dataset_name,
                         save_file=True,
                         display_visuals=True):
        """
        df:
            Pandas DataFrame object


        save_file:

        """

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        dtypes_df = pd.DataFrame({'Data Types': df.dtypes.values})
        dtypes_df.index = df.dtypes.index.tolist()
        dtypes_df.index.name = "Features"
        dtypes_df["Data Types"] = dtypes_df["Data Types"].astype(str)

        print(f"Your selected dataframe has {df.shape[1]} features.")
        if self.__notebook_mode:
            display(dtypes_df)
        else:
            if display_visuals:
                print(dtypes_df)

        if save_file:
            df_to_image(dtypes_df,
                        self.get_output_folder(),
                        f"{dataset_name}/Tables",
                        "Data Types Table",
                        show_index=True,
                        format_float_pos=2)
        plt.close()

    def missing_values_table(self,
                             df,
                             dataset_name,
                             save_file=True,
                             display_visuals=True):
        """

        df:
            Pandas DataFrame object

        Returns/Descr:
            Returns/Saves a Pandas DataFrame object giving the percentage of the
            null data for the original DataFrame columns.
        """

        if not self.__called_from_peform:
            self.__check_dataframe(df)

        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
            columns={0: 'Missing Values', 1: '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
            '% of Total Values', ascending=False).round(1)
        print(f"Your selected dataframe has {str(df.shape[1])} columns.\n"
              f"That are {str(mis_val_table_ren_columns.shape[0])} columns.\n")

        if self.__notebook_mode:
            if display_visuals:
                display(mis_val_table_ren_columns)
        else:
            if display_visuals:
                print(mis_val_table_ren_columns)

        # ---
        if save_file:
            df_to_image(mis_val_table_ren_columns,
                        self.get_output_folder(),
                        f"{dataset_name}/Tables",
                        "Missing Data Table",
                        show_index=True,
                        format_float_pos=2)

        plt.close()
