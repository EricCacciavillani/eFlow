import pandas as pd
from IPython.display import display

from eflow.utils.image_utils import df_to_image
from eflow.utils.string_utils import convert_to_filename

def data_types_table(df):
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


