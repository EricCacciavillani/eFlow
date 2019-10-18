import pandas as pd

from eflow.utils.sys_utils import write_object_text_to_file
from eflow.utils.image_utils import df_to_image


__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


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
    col_vc_df = df[feature_name].value_counts().rename_axis(
        'Unique Values').reset_index(name='Counts')

    col_vc_df["Percantage"] = ["{0:.4f}%".format(count/df.shape[0] * 100)
                               for value, count in
                               df[feature_name].value_counts().items()]

    col_vc_df.set_index('Unique Values',
                        inplace=True)

    return col_vc_df


def descr_table(df,
                feature_name):
    """
    Desc:
        Creates numerical description of a feature of a dataframe.

    Args:
        df:
            Pandas DataFrame object.

        feature_name:
            Specified feature column name.

    Returns/Descr:
        Returns back a Dataframe object of a numerical feature's summary.
    """

    col_desc_df = df[feature_name].describe().to_frame()
    col_desc_df.loc["var"] = df[feature_name].var()

    return col_desc_df


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