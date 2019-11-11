from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import correct_directory_path
from eflow.utils.sys_utils import write_object_text_to_file

import pandas as pd
from matplotlib import pyplot as plt
import copy
import six
import numpy as np

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


def df_to_image(df,
                directory_path,
                sub_dir,
                filename,
                sharpness=1.7,
                col_width=5.0,
                row_height=0.625,
                font_size=14,
                header_color='#40466e',
                row_colors=['#f1f1f2', 'w'],
                edge_color='w',
                bbox=[0, 0, 1, 1],
                header_columns=0,
                ax=None,
                show_index=False,
                index_color="#add8e6",
                format_float_pos=None,
                show_plot=False,
                **kwargs):

    directory_path = correct_directory_path(directory_path)
    df = copy.deepcopy(df)

    if format_float_pos and format_float_pos > 1:
        float_format = '{:,.' + str(2) + 'f}'
        for col_feature in set(df.select_dtypes(include=["float"]).columns):
            df[col_feature] = df[col_feature].map(float_format.format)

    if ax is None:
        size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if show_index:
        df.reset_index(inplace=True)

    mpl_table = ax.table(cellText=df.values, bbox=bbox,
                         colLabels=df.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if index_color and show_index and k[1] == 0:
                cell.set_facecolor(index_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    if not sub_dir:
        sub_dir = ""

    create_plt_png(directory_path,
                   sub_dir,
                   filename,
                   sharpness)
    if show_plot:
        plt.show()

    plt.close()


def check_if_feature_exists(df,
                            feature_name):
    """
    Desc:
        Checks if feature exists in the dataframe.

    Args:
        df:
            Pandas dataframe object.

        feature_name:
            Feature's name.

    Raises:
        Raises an error in the feature does not exist in the columns.
    """
    if feature_name not in df.columns:
        raise KeyError(
            f"The feature \'{feature_name}\' was not found in the dataframe!"
            + " Please select a valid feature from the dataframe")


def data_types_table(df,
                     sort_by_type=True):
    """
    Desc:
        Creates a pandas dataframe based on the features and their types of the
        given/passed dataframe.

    Args:
        df:
            Pandas DataFrame object

        sort_by_type:
            Orders the resulting dataframe by alphabetically by type.

    Returns:
        Returns a pandas dataframe of features and their found types
        in the passed dataframe.
    """

    if not df.shape[0]:
        print("Empty dataframe found!")
        return None

    dtypes_df = pd.DataFrame({'Data Types': df.dtypes.values})
    dtypes_df.index = df.dtypes.index.tolist()
    dtypes_df.index.name = "Features"
    dtypes_df["Data Types"] = dtypes_df["Data Types"].astype(str)

    if sort_by_type:
        dtypes_df = dtypes_df.sort_values("Data Types")

    return dtypes_df


def missing_values_table(df):
    """
    Desc:
        Creates a pandas dataframe based on the missing data inside the
        given/passed dataframe

    Args:
        df:
            Pandas DataFrame object

    Returns:
        Returns a Pandas DataFrame object giving the percentage of
        the null data for the original DataFrame columns.
    """

    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)

    return mis_val_table_ren_columns


def generate_meta_data(df,
                       output_folder_path):
    """
    Desc:
        Creates files representing the shape and feature types of the dataframe.

    Args:
        df:
            Pandas DataFrame object

        output_folder_path:
            Pre defined path to already existing directory to output file(s).

    Returns:
        Creates meta data on the passed datafrane.
    """

    # Create files relating to dataframe's shape
    shape_df = pd.DataFrame.from_dict({'Rows': [df.shape[0]],
                                       'Columns': [df.shape[1]]})
    df_to_image(shape_df,
                output_folder_path,
                "_Extras",
                "Dataframe Shape Table",
                show_index=False)

    write_object_text_to_file(shape_df.to_dict('records'),
                              output_folder_path,
                              "Dataframe Shape Text")

    # Create files relating to dataframe's types
    dtypes_df = data_types_table(df)

    df_to_image(dtypes_df,
                output_folder_path,
                "_Extras",
                "Dataframe Types Table",
                show_index=False)

    write_object_text_to_file(shape_df.to_dict('index'),
                              output_folder_path,
                              "Dataframe Feature Types Text")


def value_counts_table(df,
                       feature_name):
    """
    Desc:
        Creates a value counts dataframe.

    Args:
        df:
            Pandas DataFrame object.

        feature_name:
            Specified feature column name.

    Returns:
        Returns back a pandas Dataframe object of a feature's value counts
        with percentages.
    """

    # Value counts DataFrame
    value_count_df = df[feature_name].value_counts().rename_axis(
        'Unique Values').reset_index(name='Counts')

    total_count = sum(df[feature_name].dropna().value_counts().values)
    value_count_df["Percantage"] = ["{0:.4f}%".format(count/total_count * 100)
                                    for value, count in
                                    df[feature_name].value_counts().items()]

    value_count_df.set_index('Unique Values',
                             inplace=True)

    return value_count_df


def descr_table(df,
                feature_name,
                to_numeric=False):
    """
    Desc:
        Creates numerical description of a feature of a dataframe.

    Args:
        df:
            Pandas DataFrame object.

        feature_name:
            Specified feature column name.

        to_numeric:
            Converts the pandas series to all numeric.

    Returns/Descr:
        Returns back a Dataframe object of a numerical feature's summary.
    """

    # Convert to numeric without raising errors
    if to_numeric:
        errors = "coerce"
    else:
        errors = "ignore"

    desc_df = pd.to_numeric(df[feature_name],
                            errors=errors).dropna().describe().to_frame()
    desc_df.loc["var"] = pd.to_numeric(df[feature_name],
                                       errors=errors).dropna().var()

    return desc_df



def suggest_removal_features(df):
    """
    Args:
        df:
            Pandas DataFrame object.

    Returns:
        Returns back a list of features to remove.
    """
    features_to_remove = set()
    for feature in df.columns:
        if len(df[feature].value_counts().index.tolist()) >= int(
                df.shape[0] / 2):
            features_to_remove.add(feature)
    return features_to_remove