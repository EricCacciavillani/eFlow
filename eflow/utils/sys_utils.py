import os
import pickle
import json
from eflow.utils.string_utils import convert_to_filename, correct_directory_path


def create_unique_directory(directory_pth,
                            folder_name):
    """
    directory_pth:
        Given path that already exists.

    folder_name:
        Generated folder path.

    Returns/Desc:
        Creates a unique folder in the proper directory structure.
    """

    os.makedirs(get_unique_directory_path(directory_pth,
                                     folder_name))

def get_unique_directory_path(directory_pth,
                              folder_name):
    """
    directory_pth:
        Given path that already exists.

    folder_name:
        Generated folder path.

    Returns/Desc:
        Returns back a directory path with a unique folder name.
    """

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
    directory_pth:
        Given path that already exists.

    folder_name:
        Generated folder path.

    Returns/Desc:
        Checks/Creates required directory structures inside
        the parent directory figures.
    """

    if not os.path.exists(directory_pth):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

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
    obj:
        Any object that has a string 'repr'.

    directory_pth:
        Given path that already exists.

    filename:
        Text file's name.

    Returns/Desc:
        Writes the object to a text file.
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
    obj:
        Any python object that can be pickled.

    directory_pth:
        Given path that already exists.

    filename:
         Pickle file's name.

    Returns/Desc:
        Writes the object to a pickle file.

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

def create_json_object_from_dict(dict_obj,
                                 directory_pth,
                                 filename):
    """
    dict_obj:
        Dictionary object.

    directory_pth:
        Given path that already exists.
    filename:
        Json file's name.

    Returns/Desc:
        Writes a dict to a json file.
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

