def evaluate_model(self,
                   model_name,
                   model,
                   df,
                   df_features,
                   output_folder=None,
                   le_map=None,
                   show_extra=True,
                   find_nearest_on_cols=False,
                   zscore_low=-2,
                   zscore_high=2):
    """
    model_name:
        The string key to give the dict

    model:
        Cluster model type; it must have '.labels_' as an attribute

    df:
        Dataframe object

    df_features:
        DataFrameTypes object; organizes feature types into groups.

    output_folder:
        Sub directory to put the pngs

    le_map:
        Dict of dataframe cols to LabelEncoders

    show_extra:
        Show extra information from all functions

    find_nearest_on_cols:
            Allows columns to be converted to actual values found within
            the dataset.
            Ex: Can't have .7 in a bool column convert's it to 1.

            False: Just apply to obj columns and bool columns

            True: Apply to all columns

    zscore_low/zscore_high:
        Defines how the threshold to remove data points when profiling the
        cluster.


    The main purpose of 'evaluate_model' is to display/save tables/plots
    accoiated with describing the model's 'findings'.
    """

    df = copy.deepcopy(df)

    # Create folder structure for png outputs
    if not output_folder:
        output_path = str(model).split("(", 1)[0] + "/" + model_name
    else:
        output_path = output_folder + "/" + model_name

    # ---
    # self.__visualize_clusters(model=model,
    #                           output_path=output_path,
    #                           model_name=model_name)

    # ---
    df["Cluster_Name"] = model.labels_
    numerical_features = df_features.numerical_features()
    clustered_dataframes, shrunken_labeled_df = \
        self.__create_cluster_sub_dfs(
            df=df, model=model, numerical_features=numerical_features,
            zscore_low=zscore_low, zscore_high=zscore_high)

    rows_count, cluster_profiles_df = self.__create_cluster_profiles(
        clustered_dataframes=clustered_dataframes,
        shrunken_df=shrunken_labeled_df,
        numerical_features=df_features.numerical_features(),
        le_map=le_map,
        output_path=output_path,
        show=show_extra,
        find_nearest_on_cols=find_nearest_on_cols)

    # Check to see how many data points were removed during the profiling
    # stage
    print("Orginal points in dataframe: ", df.shape[0])
    print("Total points in all modified clusters: ", rows_count)
    print("Shrank by: ", df.shape[0] - rows_count)

    # In case to many data points were removed
    if cluster_profiles_df.shape[0] == 0:
        print(
            "The applied Z-scores caused the cluster profiles "
            "to shrink to far for the model {0}!".format(
                model_name))

    # Display and save dataframe table
    else:
        display(cluster_profiles_df)
        self.__render_mpl_table(cluster_profiles_df, sub_dir=output_path,
                                filename="All_Clusters",
                                header_columns=0, col_width=2.0)

def __create_cluster_profiles(self,
                              clustered_dataframes,
                              shrunken_df,
                              numerical_features,
                              le_map,
                              output_path,
                              find_nearest_on_cols=False,
                              show=True):
    """
        Profile each clustered dataframe based off the given mean.
        Displays extra information in dataframe tables to be understand
        each cluster.

        find_nearest_on_cols:
            Allows columns to be converted to actual values found within
            the dataset.
            Ex: Can't have .7 in a bool column convert's it to 1.

            False: Just apply to obj columns and bool columns

            True: Apply to all columns
    """

    def find_nearest(numbers, target):
        """
            Find the closest fitting number to the target number
        """
        numbers = np.asarray(numbers)
        idx = (np.abs(numbers - target)).argmin()
        return numbers[idx]

    cluster_profiles_df = pd.DataFrame(columns=shrunken_df.columns).drop(
        'Cluster_Name', axis=1)
    rows_count = 0
    for cluster_identfier, cluster_dataframe in \
            clustered_dataframes.items():
        df = pd.DataFrame(columns=cluster_dataframe.columns)
        df = df.append(cluster_dataframe.mean(), ignore_index=True)
        df.index = [cluster_identfier]

        if cluster_dataframe.shape[0] <= 1:
            continue

        # Attempt to convert numbers found within the full set of data
        for col in cluster_dataframe.columns:
            if col not in numerical_features or find_nearest_on_cols:
                df[col] = find_nearest(numbers=shrunken_df[
                    col].value_counts().index.tolist(),
                    target=df[col].values[0])

        # Evaluate cluster dataframe by dataframe
        eval_df = pd.DataFrame(columns=cluster_dataframe.columns)
        eval_df = eval_df.append(
            cluster_dataframe.mean(), ignore_index=True)
        eval_df = eval_df.append(
            cluster_dataframe.min(), ignore_index=True)
        eval_df = eval_df.append(
            cluster_dataframe.median(),
            ignore_index=True)
        eval_df = eval_df.append(
            cluster_dataframe.max(), ignore_index=True)
        eval_df = eval_df.append(
            cluster_dataframe.std(), ignore_index=True)
        eval_df = eval_df.append(
            cluster_dataframe.var(), ignore_index=True)
        eval_df.index = ["Mean", "Min", "Median",
                         "Max", "Standard Deviation", "Variance"]

        if show:
            print("Total found in {0} is {1}".format(
                cluster_identfier, cluster_dataframe.shape[0]))
            self.__render_mpl_table(
                df,
                sub_dir=output_path,
                filename=cluster_identfier +
                "_Means_Rounded_To_Nearest_Real_Numbers",
                header_columns=0,
                col_width=4.0)

            self.__render_mpl_table(
                eval_df,
                sub_dir=output_path,
                filename=cluster_identfier +
                "_Eval_Df",
                header_columns=0,
                col_width=4.0)
            display(df)
            display(eval_df)
            self.__vertical_spacing(7)

        cluster_profiles_df = cluster_profiles_df.append(
            self.__decode_df(df, le_map))

        rows_count += cluster_dataframe.shape[0]

    return rows_count, cluster_profiles_df
