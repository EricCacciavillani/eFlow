import pandas as pd

from eflow.utils.sys_utils import write_object_text_to_file
from eflow.utils.image_utils import df_to_image


def check_if_feature_exists(df,
                            feature_name):
    if feature_name not in df.columns:
        raise KeyError(
            f"The feature \'{feature_name}\' was not found in the dataframe!"
            + " Please select a valid feature from the dataframe")


def data_types_table(df,
                     sort_by_type=True):
    """
    df:
        Pandas DataFrame object

    Returns/Desc:
        Returns a pandas dataframe of features and their found types
        in the passed dataframe.
    """

    if not df.shape[0]:
        print("Empty dataframe found! This function requires a dataframe"
              "in both rows and columns.")
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
    df:
        Pandas DataFrame object

    Returns/Descr:
            Returns/Saves a Pandas DataFrame object giving the percentage of
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

    df:
        Pandas DataFrame object

    output_folder_path:
        Pre defined path to already existing directory to output file(s).

    Returns/Desc:
        Creates meta data on the passed directory ### FUTURE ERIC COME BACK AND FINALIZE THE REST ASSHAT*
    """

    shape_df = pd.DataFrame.from_dict({'Rows': [df.shape[0]],
                                       'Columns': [df.shape[1]]})
    df_to_image(shape_df,
                output_folder_path,
                "_Extras",
                "Dataframe Shape Table",
                show_index=False)

    dtypes_df = data_types_table(df)

    df_to_image(dtypes_df,
                output_folder_path,
                "_Extras",
                "Dataframe Types Table",
                show_index=False)

    write_object_text_to_file({'Rows': [df.shape[0]],
                               'Columns': [df.shape[1]]},
                              output_folder_path,
                              "Dataframe Shape Text")


def value_counts_table(df,
                       feature_name):
    """
    df:
        Pandas DataFrame object.

    feature_name:
        Specified feature column name.

    Returns/Descr:
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
    df:
        Pandas DataFrame object.

    Returns/Desc:
        Returns back a list of features to remove.
    """
    features_to_remove = set()
    for feature in df.columns:
        if len(df[feature].value_counts().index.tolist()) >= int(
                df.shape[0] / 2):
            features_to_remove.add(feature)
    return features_to_remove