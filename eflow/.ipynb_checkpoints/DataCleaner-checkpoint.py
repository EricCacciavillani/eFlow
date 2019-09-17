import numpy as np
import pandas as pd

class DataCleaner:

    def __init__(self,
                 df=None,
                 df_features=None,
                 project_name="Default_Project_Name_Data_Cleaner",
                 overwrite_full_path=None,
                 notebook_mode=True,
                 missing_data_graphing=True):
        DataAnalysis