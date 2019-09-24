import copy

def create_decorrelate_df(df,
                          df_features,
                          target_name,
                          desired_col_average=0.5,
                          show=True):
    df = copy.deepcopy(df)
    df_features = copy.deepcopy(df_features)
    while True:

        # Display correlation map
        corr_metrics = df.corr()
        if show and PROJECT.IS_NOTEBOOK:
            display(corr_metrics.style.background_gradient())

        # Get the correlation means of each feature
        corr_feature_means = []
        for feature_name in list(corr_metrics.columns):

            # Ignore target feature; Only a problem if target was numerical
            if target_name != feature_name:
                corr_feature_means.append(corr_metrics[feature_name].mean())

        if show:
            # Display graph rank
            display_rank_graph(feature_names=list(corr_metrics.columns),
                               metric=corr_feature_means,
                               title="Average Feature Correlation",
                               y_title="Correlation Average",
                               x_title="Features")

        index, max_val = get_max_index_val(corr_feature_means)

        if max_val > desired_col_average:
            # Drop col and notify
            feature_name = list(corr_metrics.columns)[index]
            df.drop(feature_name, axis=1, inplace=True)
            df_features.remove(feature_name)
            print("Dropped column: {0}".format(feature_name))
            vertical_spacing(5)

        # End loop desired average reached
        else:
            if show and PROJECT.IS_NOTEBOOK:
                display(corr_feature_means)
            break

    return df, df_features


def random_partition_of_random_samples(list_of_df_indexes,
                                       random_sampled_rows,
                                       random_sample_amount,
                                       random_state=None):
    # Convert to numpy array if list
    if isinstance(list_of_df_indexes, list):
        list_of_df_indexes = np.array(list_of_df_indexes)

    np.random.seed(random_state)
    for _ in range(np.random.randint(1, 3)):
        np.random.shuffle(list_of_df_indexes)

    if random_sample_amount > len(list_of_df_indexes):
        random_sample_amount = len(list_of_df_indexes)

    return_matrix = np.zeros((random_sample_amount, random_sampled_rows))
    for i in range(random_sampled_rows):
        sub_list = list_of_df_indexes[:random_sample_amount]
        return_matrix[i] = sub_list
        np.random.shuffle(list_of_df_indexes)

    np.random.seed(None)
    return return_matrix


def kmeans_impurity_sample_removal(df,
                                   target,
                                   pca_perc,
                                   majority_class,
                                   majority_class_threshold=.5,
                                   random_state=None):
    """
        Generate models based on the found 'elbow' of the interia values.
    """
    df = copy.deepcopy(df)
    removal_df_indexes = []

    # Create scaler object
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df.drop(columns=[target]))

    print("\nInspecting scaled results!")
    # self.__inspect_feature_matrix(matrix=scaled,
    #                               feature_names=df.columns)

    pca = PCA()
    scaled = pca.fit_transform(scaled)

    # Generate "dummy" feature names
    pca_feature_names = ["PCA_Feature_" +
                         str(i) for i in range(1,
                                               len(df.columns) + 1)]

    print("\nInspecting applied pca results!")
    # self.__inspect_feature_matrix(matrix=scaled,
    #                               feature_names=pca_feature_names)

    if pca_perc < 1.0:
        # Find cut off point on cumulative sum
        cutoff_index = np.where(
            pca.explained_variance_ratio_.cumsum() > pca_perc)[0][0]
    else:
        cutoff_index = scaled.shape[1] - 1

    print(
        "After applying pca with a cutoff percentage {0}%"
        " for the cumulative index. Using features 1 to {1}".format(
            pca_perc, cutoff_index + 1))

    print("Old shape {0}".format(scaled.shape))

    scaled = scaled[:, :cutoff_index + 1]
    pca_feature_names = pca_feature_names[0: cutoff_index + 1]

    print("New shape {0}".format(scaled.shape))

    scaled = scaler.fit_transform(scaled)

    print("\nInspecting re-applied scaled results!")
    # self.__inspect_feature_matrix(matrix=scaled,
    #                               feature_names=pca_feature_names)

    ks = range(1, 15)
    inertias = []
    all_models = []

    for k in tqdm(ks):
        # Create a KMeans instance with k clusters: model
        model = KMeans(n_clusters=k,
                       random_state=random_state).fit(scaled)

        # Append the inertia to the list of inertias
        inertias.append(model.inertia_)
        all_models.append(model)

    a = KneeLocator(inertias, ks, curve='convex', direction='decreasing')
    elbow_index = np.where(inertias == a.knee)
    best_worst_model_index = elbow_index[0][0] + 2
    best_worst_model = None

    if len(all_models) < best_worst_model_index:
        print("OUT OF INDEX ERROR!!!")
    else:
        best_worst_model = all_models[best_worst_model_index]

    print(best_worst_model.labels_)
    df["Cluster_Name"] = model.labels_
    for val in set(df["Cluster_Name"]):
        sub_df = df[df["Cluster_Name"] == val]
        val_counts_dict = sub_df[target].value_counts().to_dict()

        if majority_class in val_counts_dict and val_counts_dict[
            majority_class] / sum(val_counts_dict.values()) <= majority_class_threshold:
            removal_df_indexes += sub_df[
                sub_df[target] == majority_class].index.values.tolist()

    return removal_df_indexes
