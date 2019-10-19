import os
import pickle
import json
from eflow.utils.string_utils import convert_to_filename, correct_directory_path

__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

def create_unique_directory(directory_pth,
                            folder_name):
    """
    Desc:
        Creates a unique folder in the proper directory structure.

    Args:
        directory_pth:
            Given path that already exists.

        folder_name:
            Folder name to generated.
    """

    os.makedirs(get_unique_directory_path(directory_pth,
                                          folder_name))

def get_unique_directory_path(directory_pth,
                              folder_name):
    """
    Desc:
        Iterate through directory structure until a unique folder name can be
        found.

        Note:
            Keeps changing the folder name by appending 1 each iteration.

    Args:
        directory_pth:
            Given path that already exists.

        folder_name:
             Given path that already exists.

    Returns:
        Returns back a directory path with a unique folder name.
    """

    # -----
    directory_pth = correct_directory_path(directory_pth)
    if not os.path.exists(directory_pth):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

    check_create_dir_structure(directory_pth=directory_pth,
                               create_sub_dir="")

    # Ensures the folder is unique in the directory
    iterable = 0
    while True:
        if iterable != 0:
            created_path = f'{directory_pth}{folder_name} {iterable}'
        else:
            created_path = f'{directory_pth}{folder_name}'

        if not os.path.exists(created_path):
            break

        iterable += 1

    return created_path


def check_create_dir_structure(directory_pth,
                               create_sub_dir):
    """
    Desc:
        Checks/Creates required directory structures inside the parent
        directory figures.

    Args:
        directory_pth:
            Given path that already exists.

        create_sub_dir:
            Sub directory to create a given folder path.

    Returns:
        Returns back the created directory/
    """

    if not os.path.exists(directory_pth):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are not created "
                          "you must have a pre-defined path.")

    directory_pth = correct_directory_path(directory_pth)

    for dir in create_sub_dir.split("/"):
        directory_pth += "/" + dir
        if not os.path.exists(directory_pth):
            os.makedirs(directory_pth)

    return correct_directory_path(directory_pth)


def write_object_text_to_file(obj,
                              directory_pth,
                              filename):
    """
    Desc:
        Writes the object's string representation to a text file.

    Args:
        obj:
            Any object that has a string 'repr'.

        directory_pth:
            Given path that already exists.

        filename:
            Text file's name.
    """
    directory_pth = correct_directory_path(directory_pth)
    if not os.path.exists(directory_pth):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

    # Ensures no file extensions in filename
    filename = filename.split(".")[0]


    file_dir = f'{directory_pth}{convert_to_filename(filename)}.txt'

    f = open(file_dir, 'w')
    f.write('obj = ' + repr(obj) + '\n')
    f.close()

def pickle_object_to_file(obj,
                          directory_pth,
                          filename):
    """
    Desc:
        Writes the object to a pickle file.

    Args:
        obj:
            Any python object that can be pickled.

        directory_pth:
            Given path that already exists.

        filename:
             Pickle file's name.
    """
    directory_pth = correct_directory_path(directory_pth)
    if not os.path.exists(directory_pth):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

    # Ensures no file extensions in filename
    filename = filename.split(".")[0]
    file_dir = f'{directory_pth}{convert_to_filename(filename)}.pkl'
    list_pickle = open(file_dir, 'wb')
    pickle.dump(obj,
                list_pickle)
    list_pickle.close()

def create_json_file_from_dict(dict_obj,
                               directory_pth,
                               filename):
    """
    Desc:
        Writes a dict to a json file.

    Args:
        dict_obj:
            Dictionary object.

        directory_pth:
            Given path that already exists.

        filename:
            Json file's name.
    """
    directory_pth = correct_directory_path(directory_pth)
    if not os.path.exists(directory_pth):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")
    with open(f'{directory_pth}{convert_to_filename(filename)}.json',
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

def get_all_directories_from_path(directory_pth):
    """
    Desc:
       Gets directories names with the provided path.

    Args:
        directory_pth:
            Given path that already exists.

    Returns:
        Returns back a set a directories with the provided path.
    """

    dirs_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_pth):
        dirs_in_paths.extend(dirnames)
        break

    return set(dirs_in_paths)


def get_all_files_from_path(directory_pth,
                            file_extension=None):
    """
    Desc:
        Gets all filenames with the provided path.

    Args:
        directory_pth:
            Given path that already exists.

        file_extension:
            Only return files that have a given extension.

    Returns:
        Returns back a set a filenames with the provided path.
    """

    files_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_pth):

        if file_extension:
            file_extension = file_extension.replace(".","")
            for file in filenames:
                if file.endswith(f'.{file_extension}'):
                    files_in_paths.append(file)
        else:
            files_in_paths.extend(filenames)
        break

    return set(files_in_paths)
