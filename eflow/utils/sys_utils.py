from eflow.utils.string_utils import convert_to_filename, \
    correct_directory_path
from eflow._hidden.constants import SYS_CONSTANTS
import os
import pickle
import json
import shutil

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

def check_if_directory_exists(directory_path):
    """
    Desc:
        Checks if the given directory path exists. Raises an error if doesn't

    Args:
        directory_path:
            Given path that already exists.

    Raises:
        Raise an error if the given directory does not exist on the users system.
    """
    if not os.path.exists(directory_path):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

def create_unique_directory(directory_path,
                            folder_name):
    """
    Desc:
        Creates a unique folder in the proper directory structure.

    Args:
        directory_path:
            Given path that already exists.

        folder_name:
            Folder name to generated.
    """

    os.makedirs(get_unique_directory_path(directory_path,
                                          folder_name))

def get_unique_directory_path(directory_path,
                              folder_name):
    """
    Desc:
        Iterate through directory structure until a unique folder name can be
        found.

        Note:
            Keeps changing the folder name by appending 1 each iteration.

    Args:
        directory_path:
            Given path that already exists.

        folder_name:
             Folder name to compare against other directories that exist in the
             directory_path.

    Returns:
        Returns back a directory path with a unique folder name.
    """

    # -----
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    create_dir_structure(directory_path=directory_path,
                         create_sub_dir="")

    # Ensures the folder is unique in the directory
    iterable = 0
    while True:
        if iterable != 0:
            created_path = f'{directory_path}{folder_name} {iterable}'
        else:
            created_path = f'{directory_path}{folder_name}'

        if not os.path.exists(created_path):
            break

        iterable += 1

    return created_path


def create_dir_structure(directory_path,
                         create_sub_dir):
    """
    Desc:
        Creates required directory structures inside the parent
        directory figures.

    Args:
        directory_path:
            Given path that already exists.

        create_sub_dir:
            Sub directory to create a given folder path.

    Returns:
        Returns back the created directory.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    for dir in create_sub_dir.split("/"):
        directory_path += "/" + dir
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    return correct_directory_path(directory_path)


def write_object_text_to_file(obj,
                              directory_path,
                              filename):
    """
    Desc:
        Writes the object's string representation to a text file.

    Args:
        obj:
            Any object that has a string 'repr'.

        directory_path:
            Given path that already exists.

        filename:
            Text file's name.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    # Ensures no file extensions in filename
    filename = filename.split(".")[0]


    file_dir = f'{directory_path}{convert_to_filename(filename)}.txt'

    f = open(file_dir, 'w')
    f.write('obj = ' + repr(obj) + '\n')
    f.close()

def pickle_object_to_file(obj,
                          directory_path,
                          filename):
    """
    Desc:
        Writes the object to a pickle file.

    Args:
        obj:
            Any python object that can be pickled.

        directory_path:
            Given path that already exists.

        filename:
             Pickle file's name.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    # Ensures no file extensions in filename
    filename = filename.split(".")[0]
    file_dir = f'{directory_path}{convert_to_filename(filename)}.pkl'
    list_pickle = open(file_dir, 'wb')
    pickle.dump(obj,
                list_pickle)
    list_pickle.close()

def create_json_file_from_dict(dict_obj,
                               directory_path,
                               filename):
    """
    Desc:
        Writes a dict to a json file.

    Args:
        dict_obj:
            Dictionary object.

        directory_path:
            Given path that already exists.

        filename:
            Json file's name.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    with open(f'{directory_path}{convert_to_filename(filename)}.json',
              'w',
              encoding='utf-8') as outfile:
        json.dump(dict_obj,
                  outfile,
                  ensure_ascii=False,
                  indent=2)

def json_file_to_dict(filepath):
    """
    Desc:
        Returns back the dictionary from of a json file.

    Args:
        filepath:
             Given path to the filename.

    Returns/Desc:
        Returns back the dictionary from of a json file.
    """
    json_file = open(filepath)
    json_str = json_file.read()
    json_data = json.loads(json_str)

    return json_data

def get_all_directories_from_path(directory_path):
    """
    Desc:
       Gets directories names with the provided path.

    Args:
        directory_path:
            Given path that already exists.

    Returns:
        Returns back a set a directories with the provided path.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    dirs_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_path):
        dirs_in_paths.extend(dirnames)
        break

    return set(dirs_in_paths)


def get_all_files_from_path(directory_path,
                            file_extension=None):
    """
    Desc:
        Gets all filenames with the provided path.

    Args:
        directory_path:
            Given path that already exists.

        file_extension:
            Only return files that have a given extension.

    Returns:
        Returns back a set a filenames with the provided path.
    """
    directory_path = correct_directory_path(directory_path)
    check_if_directory_exists(directory_path)

    files_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_path):

        if file_extension:
            file_extension = file_extension.replace(".","")
            for file in filenames:
                if file.endswith(f'.{file_extension}'):
                    files_in_paths.append(file)
        else:
            files_in_paths.extend(filenames)
        break

    return set(files_in_paths)


def move_folder_to_eflow_garbage(directory_path,
                                 create_sub_dir=None):
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

