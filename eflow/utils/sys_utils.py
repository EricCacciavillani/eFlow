import os
import pickle
import json
from eflow.utils.string_utils import convert_to_filename, correct_directory_path


def create_unique_directory(directory_pth,
                            folder_name):

    os.makedirs(get_unique_directory_path(directory_pth,
                                     folder_name))

def get_unique_directory_path(directory_pth,
                              folder_name):

    directory_pth = correct_directory_path(directory_pth)
    check_create_dir_structure(directory_pth=directory_pth,
                               sub_dir="")
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
                               sub_dir):
    """
        Checks/Creates required directory structures inside
        the parent directory figures.
    """
    directory_pth = correct_directory_path(directory_pth)

    for dir in sub_dir.split("/"):
        directory_pth += "/" + dir
        if not os.path.exists(directory_pth):
            os.makedirs(directory_pth)

    return directory_pth


def write_object_text_to_file(obj,
                              directory_pth,
                              filename):
    filename = filename.split(".")[0]


    file_dir = f'{correct_directory_path(directory_pth)}{convert_to_filename(filename)}.txt'

    f = open(file_dir, 'w')
    f.write('obj = ' + repr(obj) + '\n')
    f.close()

def pickle_object_to_file(obj,
                          directory_pth,
                          filename):
    filename = filename.split(".")[0]
    file_dir = f'{correct_directory_path(directory_pth)}{convert_to_filename(filename)}.pkl'
    list_pickle = open(file_dir, 'wb')
    pickle.dump(obj,
                list_pickle)
    list_pickle.close()

def create_json_object_from_dict(dict_obj,
                                 directory_pth,
                                 filename):
    with open(f'{correct_directory_path(directory_pth)}{convert_to_filename(filename)}.json',
              'w',
              encoding='utf-8') as outfile:
        json.dump(dict_obj,
                  outfile,
                  ensure_ascii=False,
                  indent=2)

