from eflow.foundation import DataFrameTypes
from eflow.utils.sys_utils import create_dir_structure, check_if_directory_exists, \
    correct_directory_path, get_all_files_from_path, get_unique_directory_path, \
    get_all_directories_from_path, json_file_to_dict
import os
from eflow._hidden.constants import SYS_CONSTANTS
import shutil


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


def remove_unconnected_pipeline_segments():
    """
    Desc:
        Removes all pipeline segments that aren't connected to a pipeline structure.
    """

    # Directory to pipeline section
    pipeline_struct_dir = os.getcwd() + f"/{SYS_CONSTANTS.PARENT_OUTPUT_FOLDER_NAME}/_Extras/Pipeline Structure/"

    # Get all segment files by their types.
    all_segment_dirs = get_all_directories_from_path(
        pipeline_struct_dir + "/Data Pipeline Segments")
    segment_dict = dict()

    for segment_type in all_segment_dirs:
        segment_dict[segment_type] = get_all_files_from_path(
            pipeline_struct_dir + f"/Data Pipeline Segments/{segment_type}")

    # Get all segments related to each pipeline.
    all_pipeline_dirs = get_all_directories_from_path(
        pipeline_struct_dir + "/Data Pipeline/")
    pipeline_segments_dict = dict()

    for pipeline_name in all_pipeline_dirs:
        json_file = json_file_to_dict(
            f"{pipeline_struct_dir}/Data Pipeline/{pipeline_name}/root_pipeline.json")

        for i in range(1, json_file["Pipeline Segment Count"] + 1):
            segment_id = json_file["Pipeline Segment Order"][str(i)][
                'Pipeline Segment ID']
            segment_type = json_file["Pipeline Segment Order"][str(i)][
                'Pipeline Segment Type']

            if segment_type not in pipeline_segments_dict.keys():
                pipeline_segments_dict[segment_type] = {segment_id + ".json"}
            else:
                pipeline_segments_dict[segment_type].add(segment_id + ".json")

    # Remove given segments
    for segment_type, segment_ids in pipeline_segments_dict.items():
        if segment_type in segment_dict[segment_type]:
            for _id in segment_ids:
                segment_dict[segment_type].remove(_id)

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
                file_to_remove = file_to_remove.split(".")[0] + f"_{i}.json"
                i += 1

            os.rename(
                pipeline_struct_dir + f"Data Pipeline Segments/{segment_type}/{_id}",
                pipeline_struct_dir + f"Data Pipeline Segments/{segment_type}/{file_to_remove}")
            shutil.move(
                pipeline_struct_dir + f"Data Pipeline Segments/{segment_type}/{file_to_remove}",
                garbage_folder_path + file_to_remove)

