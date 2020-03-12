from eflow.utils.image_processing_utils import create_plt_png
from eflow.utils.string_utils import correct_directory_path
from eflow.utils.sys_utils import write_object_text_to_file, create_dir_structure, pickle_object_to_file
from eflow._hidden.custom_exceptions import UnsatisfiedRequirments
from eflow.utils.math_utils import calculate_entropy

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
                col_width=8,
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

    if format_float_pos and format_float_pos >= 1:
        float_format = '{:,.' + str(format_float_pos) + 'f}'
        for col_feature in set(df.select_dtypes(include=["float"]).columns):
            df[col_feature] = df[col_feature].map(float_format.format)

    if ax is None:
        size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if show_index:
        df.reset_index(inplace=True)

    mpl_table = ax.table(cellText=df.values,
                         bbox=bbox,
                         colLabels=df.columns,
                         **kwargs)

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

    plt.close("all")


def average_feature_correlation_table(df):
    """
    Desc:
        Creates a correlation table of the feature's relationship base on the
        average correlation with other features.

    Args:
        df: pd.Dataframe
            Pandas dataframe

    Returns:
        A dataframe that shows the average correlations between each feature.
    """

    corr_metrics = df.corr()
    for index, feature_index in enumerate(corr_metrics.index.tolist()):
        corr_metrics.loc[feature_index][index] = np.nan

    corr_feature_means = dict()
    for feature_name in corr_metrics.columns:
        corr_feature_means[feature_name] = corr_metrics[
            feature_name].dropna().mean()

    corr_feature_means = pd.DataFrame.from_dict(corr_feature_means,
                                                orient='index',
                                                columns=["Average Correlations"])
    corr_feature_means.index.name = "Features"

    return corr_feature_means


def feature_correlation_table(df):
    """
    Desc:
        Creates a correlation table of each feature's relationship with one
        another.

    Args:
        df: pd.Dataframe
            Pandas dataframe

    Returns:
        A dataframe that shows the correlations between each feature.
    """
    feature_corr_dict = dict()
    df_corr = df.corr()

    remaining_features = df_corr.columns.tolist()
    for main_feature in df_corr.columns:
        for sub_feature in remaining_features:
            if main_feature != sub_feature:
                feature_corr_dict[f"{main_feature} to {sub_feature}"] = \
                df_corr[main_feature].loc[sub_feature]
        remaining_features.remove(main_feature)

    df_corr = pd.DataFrame.from_dict(feature_corr_dict,
                                     orient='index',
                                     columns=["Correlations"])
    df_corr.index.name = "Features"

    return df_corr


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

    mis_val_table_ren_columns.index.name = "Features"

    return mis_val_table_ren_columns


def generate_meta_data(df,
                       output_folder_path,
                       sub_dir):
    """
    Desc:
        Creates files representing the shape and feature types of the dataframe.

    Args:
        df: pd.Dataframe
            Pandas DataFrame object

        output_folder_path: str
            Pre defined path to already existing directory to output file(s).

        sub_dir: str
            Path to be possibly generated.

    Returns:
        Creates meta data on the passed datafrane.
    """
    create_dir_structure(output_folder_path,
                         correct_directory_path(sub_dir + "/Meta Data"))

    output_folder_path = correct_directory_path(output_folder_path)

    # Create files relating to dataframe's shape
    shape_df = pd.DataFrame.from_dict({'Rows': [df.shape[0]],
                                       'Columns': [df.shape[1]]})

    if shape_df.shape[0]:
        df_to_image(shape_df,
                    f"{output_folder_path}/{sub_dir}",
                    "Meta Data",
                    "Dataframe Shape Table",
                    show_index=False)

    write_object_text_to_file(shape_df.to_dict('records'),
                              f"{output_folder_path}/{sub_dir}/Meta Data",
                              "Dataframe Shape Text")

    # Create files relating to dataframe's types
    dtypes_df = data_types_table(df)
    if dtypes_df.shape[0]:
        df_to_image(dtypes_df,
                    f"{output_folder_path}/{sub_dir}",
                    "Meta Data",
                    "Dataframe Types Table",
                    show_index=True)

    plt.close("all")


    # Missing value table
    mis_val_table = missing_values_table(df)
    if mis_val_table.shape[0]:
        df_to_image(mis_val_table,
                    f"{output_folder_path}/{sub_dir}",
                    "Meta Data",
                    "Missing Data Table",
                    show_index=True)

    plt.close("all")

def generate_entropy_table(df,
                           df_features,
                           output_folder_path,
                           sub_dir,
                           file_name="Entropy Table"):
    """
    Desc:
        Calculate the entropy of each non-continous numerical feature in a pandas
        dataframe object and store in a pandas dataframe object in the proper
        directory structure.

    Args:
        df: pd.Dataframe
            Pandas DataFrame object

        df_features: DataFrameTypes from eflow
            DataFrameTypes object

        output_folder_path: str
            Pre defined path to already existing directory to output file(s).

        sub_dir: str
            Path to be possibly generated.

        file_name: str
            Name of the given file to save

    Returns:
        Nothing
    """
    entropy_dict = dict()
    for feature_name in df.columns:
        if feature_name in df_features.all_features() and \
                feature_name not in df_features.null_only_features() and \
                feature_name not in df_features.continuous_numerical_features():
            entropy_dict[feature_name] = calculate_entropy(
                df[feature_name].dropna())

    entropy_table = pd.DataFrame.from_dict(entropy_dict,
                                           orient='index').rename(columns={0: "Entropy"})

    entropy_table.index.name = "Features"

    entropy_table.sort_values(by=["Entropy"],
                              ascending=True,
                              inplace=True)

    create_dir_structure(output_folder_path,
                         sub_dir)


    pickle_object_to_file(entropy_table,
                          output_folder_path + sub_dir,
                          file_name)

    df_to_image(entropy_table,
                output_folder_path,
                sub_dir,
                "Entropy Table",
                show_index=True,
                format_float_pos=5)


def auto_binning(df,
                 df_features,
                 feature_name,
                 bins=5):
    """
    Desc:
        Takes a pandas series object and assigns generalized labels and binning
        dimensions.

    Args:
        df: pd.Dataframe
            Pandas Datafrane object

        df_features: DataFrameTypes from eflow
            DataFrameTypes object

        feature_name: string
            Name of the feature to extract the series from

        bins: int
            Number of bins to create.

    Returns:
        Gives back the bins and associated labels
    """

    if feature_name not in df_features.all_features():
        raise KeyError("Feature name must be encapsulated in df_features.")

    if feature_name not in df_features.float_features() and feature_name not in df_features.integer_features():
        raise ValueError("Feature must be a float or an integer to properly bin the given data.")

    # Create bins of type pandas.Interval
    binned_list = list(pd.cut(df[feature_name].dropna().sort_values(),
                              bins).unique())

    # Iterate through all possible bins
    bins = []
    labels = []
    for bin_count, binned_obj in enumerate(binned_list):

        # Extract from pandas.Interval into a list; just nicer to read
        binned_obj = [binned_obj.left, binned_obj.right]

        # Convert to int if the feature is an int type
        if feature_name in df_features.integer_features():
            binned_obj[0] = int(binned_obj[0])
            binned_obj[1] = int(binned_obj[1])

        # -----
        if bin_count == 0:

            # Move bined value down so it properly captures the starting integer
            if feature_name in df_features.integer_features():
                bins.append(int(binned_obj[0]) - .000001)
            else:
                bins.append(binned_obj[0])

        # -----
        if feature_name in df_features.integer_features():

            # Values are the same change label look
            if bin_count != 0 and binned_obj[0] + 1 == binned_obj[1]:
                labels.append("=" + str(binned_obj[1]))
            else:

                if bin_count == 0:
                    labels.append(
                        str(binned_obj[0]) + u" \u27f7 " + str(binned_obj[1]))
                else:
                    labels.append(str(binned_obj[0] + 1) + u" \u27f7 " + str(
                        binned_obj[1]))

            # Move bined value up so it properly captures the ending integer
            binned_obj[1] = int(binned_obj[1]) + .000001
        else:
            labels.append(
                str(binned_obj[0]) + "+ " u"\u27f7 " + str(binned_obj[1]))

        bins.append(binned_obj[1])

    bins = [float(bins[i]) for i in range(0, len(bins))]
    return bins, labels



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
        df: pd.Dataframe
            Pandas DataFrame object.

        feature_name: string
            Specified feature column name.

        to_numeric: bool
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
    Desc:
        Will find features that appear to be almost index like with their
        feature values.
    Args:
        df: pd.Dataframe
            Pandas DataFrame object.

    Returns:
        Returns back a list of features to remove.
    """
    features_to_remove = set()

    # Return back index like features
    for feature in df.columns:
        if len(df[feature].value_counts().index.tolist()) >= int(
                df.shape[0] / 2):
            features_to_remove.add(feature)

    return features_to_remove