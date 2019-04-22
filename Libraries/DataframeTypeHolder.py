class DataframeTypeHolder:

    """
        Seperates the features based off of dtypes
        to better keep track of feature changes over time.
        Should only be used for manipulation of features.
    """

    def __init__(self, df):

        self.__bool_features = set(df.select_dtypes(include=["bool"]).columns)
        self.__categorical_features = set(
            df.select_dtypes(include=["object"]).columns)
        self.__integer_features = set(df.select_dtypes(include=["int"]).columns)
        self.__float_features = set(df.select_dtypes(include=["float"]).columns)
        self.__numerical_features = self.__float_features | self.__integer_features

    # --- Getters
    def get_numerical_features(self):
        return list(self.__numerical_features)

    def get_integer_features(self):
        return list(self.__integer_features)

    def get_float_features(self):
        return list(self.__float_features)

    def get_categorical_features(self):
        return list(self.__categorical_features)

    def get_bool_features(self):
        return list(self.__float_features)

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
        print("Categorical Features: {0}\n".format(
            self.__categorical_features))
        print("Bool Features: {0}\n".format(
            self.__bool_features))
        print("---------"*10)
        print("Numerical Features: {0}\n".format(
            self.__numerical_features))
        print("Integer Features: {0}\n".format(
            self.__integer_features))
        print("Float Features: {0}\n".format(
            self.__float_features))