import copy
from IPython.display import display, HTML


class DataFrameTypes:

    """
        Seperates the features based off of dtypes
        to better keep track of feature changes over time.
        Should only be used for manipulation of features.
    """

    def __init__(self,
                 df,
                 target_col=None,
                 ignore_nulls=True,
                 display_init=True):

        if ignore_nulls and df.isnull().values.any():
            tmp_df = copy.deepcopy(df)
            nan_columns = [feature for feature, nan_found in
                           tmp_df.isna().any().items() if nan_found]
            self.__make_type_assertion_after_ignore_nan(tmp_df,
                                                        nan_columns)
        else:
            tmp_df = df

        self.__bool_features = set(tmp_df.select_dtypes(include=["bool"]).columns)
        self.__categorical_features = set(
            tmp_df.select_dtypes(include=["object"]).columns)
        self.__integer_features = set(tmp_df.select_dtypes(include=["int"]).columns)
        self.__float_features = set(tmp_df.select_dtypes(include=["float"]).columns)
        self.__numerical_features = self.__float_features | self.__integer_features
        self.__target_feature = None
        self.__one_hot_encoded_names = dict()

        if target_col:
            if target_col in tmp_df.columns:
                self.__target_feature = target_col
            else:
                print("WARNING!!!: THE FEATURE {0} "
                      "DOES NOT EXIST IN THIS DATASET!".format(target_col))

        if self.__categorical_features:
            for col_feature in self.__categorical_features:
                
                if col_feature is not self.__target_feature: 
                    self.__one_hot_encoded_names[col_feature] = list()
                    for value_of_col in set(tmp_df[col_feature].values):
                        if isinstance(value_of_col,str):
                            self.__one_hot_encoded_names[col_feature].append(
                                col_feature + "_" + value_of_col.replace(" ",
                                                                         "_"))
        if display_init:
            self.display_all()
        features_not_captured = set(tmp_df.columns)
        for col_feature in (self.__numerical_features |
                            self.__categorical_features |
                            self.__bool_features):
            features_not_captured.remove(col_feature)

        if features_not_captured:
            print("ERROR MISSING FEATURE(S)!\n{0}".format(
                features_not_captured))

    # --- Getters
    def get_numerical_features(self,
                               exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__numerical_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__numerical_features)

    def get_integer_features(self,
                            exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__integer_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__integer_features)

    def get_float_features(self,
                           exclude_target=False):
        if exclude_target:
             return [col_feature for col_feature in self.__float_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__float_features)

    def get_categorical_features(self,
                                 exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__categorical_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__categorical_features)
            

    def get_bool_features(self,
                          exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in self.__bool_features
                    if col_feature != self.__target_feature]
        else:
            return list(self.__bool_features)
    
    def get_all_features(self,
                         exclude_target=False):
        if exclude_target:
            return [col_feature for col_feature in
                    list(self.__integer_features | self.__float_features 
                         | self.__categorical_features | self.__bool_features)
                    if col_feature != self.__target_feature]
        
        else:
            return list(self.__integer_features | self.__float_features
                        | self.__categorical_features | self.__bool_features)

    def get_target(self):
        return copy.deepcopy(self.__target_feature)

    # --- Appenders
    def append_categorical_features(self,
                                    feature_name):
        self.__categorical_features |= set(feature_name)

    def append_categorical_features(self,
                                    feature_name):
        self.__bool_features |= set(feature_name)

    def append_float_features(self,
                              feature_name):
        self.__float_features |= set(feature_name)
        self.__numerical_features |= set(feature_name)

    def append_integer_features(self,
                                feature_name):
        self.__integer_features |= set(feature_name)
        self.__numerical_features |= set(feature_name)

    def set_one_hot_encode_features(self,
                                    categorical_features):
        if not isinstance(categorical_features, list):
            categorical_features = [categorical_features]
        #
        # for col_feature in categorical_features:
        #     if col_feature in self.__categorical_features:
        #         += self.__one_hot_encoded_names[col_feature]
        #     else:
        #         print("Error")

    def set_date_columns(self):
        pass


    # ---Remover
    def remove(self,
               feature_name):
        try:
            self.__categorical_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__numerical_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__integer_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__float_features.remove(feature_name)
        except KeyError:
            pass

        try:
            self.__bool_features.remove(feature_name)
        except KeyError:
            pass

    def display_all(self):

        # Display category based features
        if self.__categorical_features:
            print("Categorical Features: {0}\n".format(
                self.__categorical_features))

        if self.__bool_features:
            print("Bool Features: {0}\n".format(
                self.__bool_features))

        if self.__one_hot_encoded_names:
            print("Possible One hot encoded feature names: {0}\n".format(
                self.__one_hot_encoded_names))

        if self.__bool_features or self.__categorical_features:
            print("---------"*10)

        # ---
        if self.__numerical_features:
            print("Numerical Features: {0}\n".format(
                self.__numerical_features))
        if self.__integer_features:
            print("Integer Features: {0}\n".format(
                self.__integer_features))

        if self.__float_features:
            print("Float Features: {0}\n".format(
                self.__float_features))

        if self.__target_feature:
            print("Target Feature: {0}\n".format(
                self.__target_feature))

    def __make_type_assertion_after_ignore_nan(self,
                                               df,
                                               nan_columns):
        float_features = set(
            df.select_dtypes(include=["float"]).columns)
        for feature in nan_columns:
            feature_values = list(df[feature].dropna().value_counts().keys())
            if len(feature_values) == 1 and (0.0 in feature_values or 1.0 in feature_values):
                df[feature] = df[feature].astype(bool)
            elif len(feature_values) == 2 and (0.0 in feature_values and 1.0 in feature_values):
                df[feature] = df[feature].astype(bool)

            elif feature in float_features:
                feature_values = [str(i) for i in feature_values]
                convert_to_float = False
                for str_val in feature_values:
                    tokens = str_val.split(".")

                    if len(tokens) > 1 and len(tokens[1]) > 1 or \
                            int(tokens[1]) > 0:
                        convert_to_float = True
                if convert_to_float:
                    df[feature].fillna(0, inplace=True)
                    df[feature] = df[feature].astype(float)
                else:
                    df[feature].fillna(0, inplace=True)
                    df[feature] = df[feature].dropna().astype(int)