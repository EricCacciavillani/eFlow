import os
import pickle
import json
from eflow.utils.string_utils import convert_to_filename, correct_directory_path


def create_unique_directory(directory_path,
                            folder_name):
    """
    directory_path:
        Given path that already exists.

    folder_name:
        Generated folder path.

    Returns/Desc:
        Creates a unique folder in the proper directory structure.
    """

    os.makedirs(get_unique_directory_path(directory_path,
                                     folder_name))

def get_unique_directory_path(directory_path,
                              folder_name):
    """
    directory_path:
        Given path that already exists.

    folder_name:
        Generated folder path.

    Returns/Desc:
        Returns back a directory path with a unique folder name.
    """

    directory_path = correct_directory_path(directory_path)
    if not os.path.exists(directory_path):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

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
    directory_path:
        Given path that already exists.

    folder_name:
        Generated folder path.

    Returns/Desc:
        Checks/Creates required directory structures inside
        the parent directory figures.
    """

    if not os.path.exists(directory_path):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are not created "
                          "you must have a pre-defined path.")

    directory_path = correct_directory_path(directory_path)

    for dir in create_sub_dir.split("/"):
        directory_path += "/" + dir
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

    return correct_directory_path(directory_path)


def write_object_text_to_file(obj,
                              directory_path,
                              filename):
    """
    obj:
        Any object that has a string 'repr'.

    directory_path:
        Given path that already exists.

    filename:
        Text file's name.

    Returns/Desc:
        Writes the object to a text file.
    """
    directory_path = correct_directory_path(directory_path)
    if not os.path.exists(directory_path):
        raise SystemError("Main directory path doesn't exist.\n"
                          "To help ensure unwanted directories are created you "
                          "must have a pre-defined path.")

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
    obj:
        Any python object that can be pickled.

    directory_path:
        Given path that already exists.

    filename:
         Pickle file's name.

    Returns/Desc:
        Writes the object to a pickle file.
    """
    directory_path = correct_directory_path(directory_path)
    if not os.path.exists(directory_path):
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

def dict_to_json_file(dict_obj,
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

def json_file_to_dict(filepath):
    """
    filepath:
         Given path to the filename.

    Returns/Desc:
        Returns back the dictionary form of a json file.
    """
    json_file = open(filepath)
    json_str = json_file.read()
    json_data = json.loads(json_str)

    return json_data

def get_all_directories_from_path(directory_pth):
    """
    directory_pth:
        Given path that already exists.

    Returns/Desc:
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
    directory_pth:
        Given path that already exists.

    file_extension:
        Only return files that have a given extension.

    Returns/Desc:
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
