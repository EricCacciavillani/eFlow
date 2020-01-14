from eflow.foundation import DataFrameTypes
from eflow.utils.sys_utils import create_dir_structure, check_if_directory_exists, \
    correct_directory_path, get_all_files_from_path, get_unique_directory_path, \
    get_all_directories_from_path, json_file_to_dict
import os
from eflow._hidden.constants import SYS_CONSTANTS
import shutil
import copy
from eflow.utils.language_processing_utils import get_synonyms


def move_folder_to_eflow_garbage(directory_path,
                                 create_sub_dir=None):
    """
    Desc:
        Renames and moves contents to a folder labeled 'Garbage' for the user/system
        to later handle.

    Args:
        directory_path:
            Path to given folder to move to 'Garbage'

        create_sub_dir:
            If the folder 'Garbage' needs further organization then you can specify
            a folder for the given folder to be embedded in.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    if not create_sub_dir:
        create_sub_dir = ""
    else:
        correct_directory_path(create_sub_dir)

    garbage_folder_path = create_dir_structure(os.getcwd(),
                                               f"{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/_Extras/Garbage/{create_sub_dir}")

    path_to_folder, folder_name = directory_path[:-1].rsplit('/', 1)

    _, folder_name = get_unique_directory_path(garbage_folder_path,folder_name).rsplit('/', 1)

    os.rename(directory_path,
              f'{path_to_folder}/{folder_name}')

    shutil.move(f'{path_to_folder}/{folder_name}', garbage_folder_path)



def get_type_holder_from_pipeline(pipeline_name):
    """
    Desc:
        Returns a type holder object from a specfied pipeline directory name.

    Args:
        pipeline_name:
            The name of the pipeline to json file to init type holder.

    Returns:
        Init type holder.
    """
    directory_path = correct_directory_path(f"{os.getcwd()}/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/_Extras/Pipeline Structure/Data Pipeline/{pipeline_name}")
    df_features = DataFrameTypes(df=None)
    directory_path = correct_directory_path(directory_path)
    df_features.init_on_json_file(directory_path + "df_features.json")
    return df_features


def create_color_dict_for_features(df,
                                   df_features,
                                   value_limit=50):
    """
    Desc:
        Attempts to give a color to each value based on the feature name and
        the feature value. This should be used as a template for the user to insert
        their own hex values.

    Note:
        Easily the messiest code I have ever written but it gets the job done...
        I am so sorry for your eyes...
    """


    '''
    First row of matrix determines the feature's name. Empty space represents
    Second row of matrix determines the value inside the feature and the associated color to that value.
    '''
    defined_feature_value_colors = list()

    defined_feature_value_colors.append([["Died", "Dead", "Deceased"],
                                         ["1", "True",
                                          "#616369"],
                                         ["0", "False",
                                          "#4dad6c"]])

    defined_feature_value_colors.append([["Survived", "Lived"],
                                         ["1", "true",
                                          "#4dad6c"],
                                         ["0", "false",
                                          "#616369"]])

    defined_feature_value_colors.append([["gender", "sex",],
                                         ["Male", "M",
                                          "#7EAED3"],
                                         ["Female", "F",
                                          "#FFB6C1"]])

    defined_feature_value_colors.append([[" "],
                                         ["y" "yes",
                                          "#55a868"],
                                         ["n", "no",
                                          "#ff8585"]])
    defined_feature_value_colors.append([[" "],
                                         ["1", "true",
                                          "#55a868"],
                                         ["0", "false",
                                          "#ff8585"]])

    all_true_vals = {"1","y","yes","true","t","y"}
    all_false_vals = {"0","n","no","false","f","n"}

    feature_value_color_dict = dict()

    # -----
    for feature_name in df_features.non_continuous_features():

        feature_values = []
        feature_value_color_dict[feature_name] = dict()
        for value in set(df[feature_name].dropna().values):
            feature_value_color_dict[feature_name][str(value)] = None
            feature_values.append(str(value))


        if len(feature_values) >= value_limit:
            del feature_value_color_dict[feature_name]
            feature_value_color_dict[feature_name] = dict()
            continue


        # Check if the given feature name matches any pre-defined names
        for defined_feature_info in copy.deepcopy(defined_feature_value_colors):

            # Extract all pre-defined feature names
            defined_feature_names = [str(x).lower()
                                     for x in defined_feature_info.pop(0)]

            # Check if any of the pre-defined feature names have synoymns with the given feature name
            feature_synonym_found = False
            for given_feature_name in defined_feature_names:
                synonyms = get_synonyms(given_feature_name)

                if feature_name in synonyms:
                    feature_synonym_found = True

            # Compare both feature names; ignore char case; check for default
            if feature_synonym_found or \
                    feature_name.lower() in defined_feature_names or \
                    " " in defined_feature_names:
                defined_feature_values = [[j.lower() for j in x]
                                          for x in defined_feature_info]

                # Finally make logical color assertions based on the feature values
                found_colors = False
                for defined_values in defined_feature_values:

                    # Check if the feature value's synonym's with pre defined feature values
                    val_synonym_found = False
                    defined_color = defined_values.pop(-1)
                    for feature_val in feature_values:

                        for _defined_val in defined_values:
                            synonyms = get_synonyms(str(_defined_val))

                            if feature_name in synonyms:
                                val_synonym_found = True

                        # -----
                        if feature_val.lower() in defined_values or val_synonym_found:
                            if feature_name in df_features.bool_features():

                                if feature_val in all_true_vals:
                                    feature_value_color_dict[feature_name][
                                        "1"] = defined_color

                                elif feature_val in all_false_vals:
                                    feature_value_color_dict[feature_name][
                                        "0"] = defined_color
                            else:
                                feature_value_color_dict[feature_name][
                                    feature_val] = defined_color

                            found_colors = True

                # Break pre-defined feature names loop
                if found_colors:
                    break

    tmp_feature_value_color_dict = copy.deepcopy(feature_value_color_dict)

    # Convert any feature values back to numeric
    for feature_name,color_value_dict in tmp_feature_value_color_dict.items():
        for feature_val, color in color_value_dict.items():

            try:
                int(feature_val)

                del feature_value_color_dict[feature_name][feature_val]

                feature_value_color_dict[feature_name][int(feature_val)] = color

            except:
                pass

    return feature_value_color_dict


def remove_unconnected_pipeline_segments():
    """
    Desc:
        Removes all pipeline segments that aren't connected to a pipeline structure.
    """


    pipeline_struct_dir = os.getcwd() + f"/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/_Extras/Pipeline Structure/"

    if not os.path.exists(pipeline_struct_dir):
        print("Project structure for pipelines has yet to be initalized. Can't clean/remove any files related to pipeline...")
    else:
        segment_dict = dict()
        pipeline_segments_dict = dict()
        # Get all segment files by their types.
        if os.path.exists(pipeline_struct_dir + "/Data Pipeline Segments/"):
            all_segment_dirs = get_all_directories_from_path(
                pipeline_struct_dir + "/Data Pipeline Segments")

            for segment_type in all_segment_dirs:
                segment_dict[segment_type] = get_all_files_from_path(
                    pipeline_struct_dir + f"/Data Pipeline Segments/{segment_type}")

        # Get all segments related to each pipeline.
        if os.path.exists(pipeline_struct_dir + "/Data Pipeline/"):
            all_pipeline_dirs = get_all_directories_from_path(
                pipeline_struct_dir + "/Data Pipeline/")

            for pipeline_name in all_pipeline_dirs:
                json_file = json_file_to_dict(
                    f"{pipeline_struct_dir}/Data Pipeline/{pipeline_name}/root_pipeline.json")

                for i in range(1, json_file["Pipeline Segment Count"] + 1):
                    segment_id = json_file["Pipeline Segment Order"][str(i)][
                        'Pipeline Segment ID']
                    segment_type = json_file["Pipeline Segment Order"][str(i)][
                        'Pipeline Segment Type']

                    if segment_type in segment_dict.keys() and segment_id + ".json" in \
                            segment_dict[segment_type]:
                        segment_dict[segment_type].remove(segment_id + ".json")

        # Create path to eflow's garbage
        garbage_folder_path = create_dir_structure(os.getcwd(),
                                                   f"{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/_Extras/Garbage/Data Pipeline Segments/DataTransformer/")

        # Rename files and move them to appropriate
        for segment_type, segment_ids in segment_dict.items():
            files_in_garbage = get_all_files_from_path(garbage_folder_path)

            for _id in segment_ids:
                file_to_remove = _id
                i = 1
                while file_to_remove in files_in_garbage:
                    file_to_remove = _id
                    file_to_remove = file_to_remove.split(".")[
                                         0] + f"_{i}.json"
                    i += 1

                os.rename(
                    pipeline_struct_dir + f"Data Pipeline Segments/{segment_type}/{_id}",
                    pipeline_struct_dir + f"Data Pipeline Segments/{segment_type}/{file_to_remove}")
                shutil.move(
                    pipeline_struct_dir + f"Data Pipeline Segments/{segment_type}/{file_to_remove}",
                    garbage_folder_path + file_to_remove)