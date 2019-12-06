# Public python libs
import itertools
import pandas as pd
import numpy as np
from collections import Counter
from collections import defaultdict
import copy
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import operator
from dtreeviz.trees import *
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as mp
import os
from IPython.display import display, HTML
import six
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
from kneed import DataGenerator, KneeLocator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import math

class ThreadSafeDict(dict) :
    def __init__(self, * p_arg, ** n_arg) :
        dict.__init__(self, * p_arg, ** n_arg)
        self._lock = threading.Lock()

    def __enter__(self) :
        self._lock.acquire()
        return self

    def __exit__(self, type, value, traceback) :
        self._lock.release()

# --- File specific constants ---
# Import personal "libs"
def enum(**enums):
    return type('Enum', (), enums)


PROJECT = enum(PATH_TO_OUTPUT_FOLDER=''.join(os.getcwd().partition('/eFlow')[0:1]) + "/Figures")


# --- Figures maintaining ---
def check_create_figure_dir(sub_dir):

    directory_pth = PROJECT.PATH_TO_OUTPUT_FOLDER

    for dir in sub_dir.split("/"):
        directory_pth += "/" + dir
        if not os.path.exists(directory_pth):
            os.makedirs(directory_pth)

    return directory_pth


def create_plt_png(sub_dir, filename):

    # Ensure directory structure is init correctly
    abs_path = check_create_figure_dir(sub_dir)

    # Ensure file ext is on the file.
    if filename[-4:] != ".png":
        filename += ".png"

    fig = plt.figure(1)
    fig.savefig(abs_path + "/" + filename, bbox_inches='tight')


def fast_eudis(v1,
               v2):
    dist = [((a - b) ** 2) * w for a, b, w in zip(v1, v2)]
    dist = math.sqrt(sum(dist))
    return dist


def rotate_point(origin,
                 point,
                 angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.

    # Author link: http://tinyurl.com/y4yz5hco
    """
    ox, oy = origin
    px, py = point
    print(px)

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy


# Not created by me!
# Author: http://tinyurl.com/y2hjhbwf
def render_mpl_table(data,
                     sub_dir,
                     filename,
                     col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    create_plt_png(sub_dir, filename)

    plt.close()


# --- General Utils/Misc ---
def enum(**enums):
    """
        Using past pythons enums b/c I like them better
    """
    return type('Enum', (), enums)


def find_nearest(numbers, target):
    """
        Find the closest fitting number to the target number
    """
    numbers = np.asarray(numbers)
    idx = (np.abs(numbers - target)).argmin()
    return numbers[idx]


def get_max_index_val(given_list):
    """
        Returns the max index and value of a list
    """
    return max(enumerate(given_list), key=operator.itemgetter(1))


# I am this lazy yes...don't hate
def vertical_spacing(spaces=1):
    """
        I could just overwrite the automatic newline builtin.
        But the code looks unclean so screw it.
    """
    for _ in range(0, spaces):
        print()


# --- Pandas dataframe maintaining ---
def replace_df_vals(passed_df, replace_dict):
    """
        Uses a dict of dict to replace dataframe data
    """

    def replace_vals_col(data, decoder):
        return decoder[data]

    df = copy.deepcopy(passed_df)
    for col in df.columns:
        if col in replace_dict.keys():
            df[col] = np.vectorize(replace_vals_col)(
                df[col], replace_dict[col])

    return df


# Returns encoded df and label encoded map
def encode_df(df, df_features):
    
    obj_cols = df_features.categorical_features()

    df = copy.deepcopy(df)
    # ---
    le_map = defaultdict(LabelEncoder)

    # Encoding the variable
    fit = df[obj_cols].apply(lambda x: le_map[x.name].fit_transform(x))

    # Inverse the encoded
    fit.apply(lambda x: le_map[x.name].inverse_transform(x))

    # Using the dictionary to label future data
    df[obj_cols] = df[obj_cols].apply(lambda x: le_map[x.name].transform(x))

    return df, le_map

def decode_df(df,le_map):
    
    df = copy.deepcopy(df)
    decode_cols = list(le_map.keys())
    df[decode_cols] = df[decode_cols].apply(lambda x: le_map[x.name].inverse_transform(x))
    
    return df


def print_encoder(le):

    """
        Display the relationship between encoded values and actual values
    """

    print("\nLabel mapping:\n")
    for i, item in enumerate(le.classes_):
        print("\t", item, '-->', i)

    # Draw a line
    print("-"*30, "\n")

def print_encoder_map(le_map):

    for column_name,le in le_map.items():
        print("\nLabel mapping:\n")

        for i, item in enumerate(le.classes_):
            print("\t", item, '-->', i)

        # Draw a line
        print("-"*30, "\n")


def remove_outliers_df(df, removal_dict):
    """
        Removes outliers with a 'High'/'Low' keyed map;
        Remove all above 'High' and all below 'Low'
    """

    df = copy.deepcopy(df)

    # Loop on columns
    for feature_name in removal_dict.keys():

        # Replacements found; remove all above 'High' and all below 'Low'
        if feature_name in df.columns:
            if removal_dict[feature_name]["High"]:
                df = df[df[feature_name] < removal_dict[feature_name]["High"]]
            elif removal_dict[feature_name]["Low"]:
                df = df[df[feature_name] > removal_dict[feature_name]["Low"]]

    return df.reset_index(drop=True)


def inspect_feature_matrix(matrix,
                           feature_names):
    """
        Creates a dataframe to quickly analyze a matrix
    """
    mean_matrix = np.mean(matrix, axis=0)
    std_matrix = np.std(matrix, axis=0)
    data_dict = dict()
    for index, feature_name in enumerate(feature_names):
        data_dict[feature_name] = [mean_matrix[index],
                                   std_matrix[index]]

    tmp_df = pd.DataFrame.from_dict(data_dict,
                                    orient='index',
                                    columns=['Mean', 'Standard Dev'])

    display(tmp_df)

    return tmp_df


# --- General Graphing ---

# Not created by me!
# Created by my teacher: Narine Hall
def visualize_pca_variance(data):
    """
        Visualize PCA matrix feature importance.
    """

    # Check for pca variance
    pca = PCA()
    pca.fit_transform(data)

    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
    plt.xticks()
    plt.ylabel('variance ratio')
    plt.xlabel('PCA feature')
    plt.tight_layout()
    plt.savefig('figures/pca_variance_ratio.png')

    plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_.cumsum())
    plt.xticks()
    plt.ylabel('cumulative sum of variances')
    plt.xlabel('PCA feature')
    plt.tight_layout()
    create_plt_png("Default/",
                   "Rank_Graph_")

    return pca

def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm,
               interpolation='nearest',
               cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def display_rank_graph(feature_names, metric,
                       title="", y_title="",
                       x_title="",
                       plt_output="Default/Analysis"):
    """
        Darker colors that have higher rankings (values)
    """
    plt.figure(figsize=(7, 7))

    # Init color ranking fo plot
    # Ref: http://tinyurl.com/ydgjtmty
    pal = sns.color_palette("GnBu_d", len(metric))
    rank = np.array(metric).argsort().argsort()
    ax = sns.barplot(y=feature_names, x=metric,
                     palette=np.array(pal[::-1])[rank])
    plt.xticks(rotation=0, fontsize=15)
    plt.yticks(fontsize=15)
    plt.xlabel(x_title, fontsize=20, labelpad=20)
    plt.ylabel(y_title, fontsize=20, labelpad=20)
    plt.title(title, fontsize=15)

    plt.show()
    plt.close()


# --- Some sklearn specific functions ---
def optimize_model_grid(model,
                        X_train,
                        y_train,
                        param_grid,
                        n_jobs,
                        cv=10):
    """
        Finds the best parameters for a grid; returns the model and parameters.
    """
    # Instantiate the GridSearchCV object
    model_cv = GridSearchCV(model, param_grid, cv=cv, n_jobs=n_jobs)

    # Fit it to the data
    model_cv.fit(X_train, y_train)

    # Print the tuned parameters and score
    print("Tuned Parameters: {}".format(model_cv.best_params_))
    print("Best score on trained data was {0:4f}".format(model_cv.best_score_))

    model = type(model)(**model_cv.best_params_)

    # Return model and parameters
    return model, model_cv.best_params_


# Not created by me!
# Author: https://github.com/scikit-learn/scikit-learn/issues/7845
def report_to_dict(cr):

    """
        Converts a 'classification_report' to a dataframe for more professional look.
    """

    # Parse rows
    tmp = list()
    for row in cr.split("\n"):
        parsed_row = [x for x in row.split("  ") if len(x) > 0]
        if len(parsed_row) > 0:
            tmp.append(parsed_row)

    # Store in dictionary
    measures = tmp[0]

    D_class_data = defaultdict(dict)
    for row in tmp[1:]:
        class_label = row[0]
        for j, m in enumerate(measures):
            D_class_data[class_label][m.strip()] = float(row[j + 1].strip())
    return D_class_data


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