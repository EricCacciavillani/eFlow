# Import libs
import os


# Taken from utils.sys_utils
def get_all_directories_from_path(directory_path):
    """
    directory_path:
        Given path that already exists.

    Returns:
        Returns back a set a directories with the provided path.
    """

    dirs_in_paths = []
    for (dirpath, dirnames, filenames) in os.walk(directory_path):
        dirs_in_paths.extend(dirnames)
        break

    return set(dirs_in_paths)


def get_all_files_from_path(directory_path,
                            file_extension=None):
    """
    directory_path:
        Given path that already exists.

    file_extension:
        Only return files that have a given extension.

    Returns:
        Returns back a set a filenames with the provided path.
    """

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

# Get the current working directory
current_work_dir = os.getcwd()
project_dir = current_work_dir  + "/eflow/"
print(project_dir)
# Get all directories from project
all_dirs = get_all_directories_from_path(project_dir)
print(all_dirs)
for dir_name in all_dirs:

    # Ignore any hidden files
    if dir_name[0] == "_":
        continue

    # Ignore utils for now
    if dir_name == "utils":
        continue

    dir_files = get_all_files_from_path(project_dir + dir_name,
                                        "py")
    print(dir_files)
    i = 0
    for file_name in dir_files:

        print(file_name)
        # Ignore hidden files
        if file_name[0] == "_":
            continue

        with open(f'{project_dir}{dir_name}/{file_name}') as fp:

            function_def = ""
            line = fp.readline()
            while line:
                if ("def " in line and "def _" not in line):
                    def_start = True
                    function_def += line.split("def ")[1].replace("(self,","(").strip()
                    while line:
                        line = fp.readline()
                        
                        if "):" in line:
                            def_start = False
                            break
                        else:
                            line = line.strip()
                            function_def += line
                    function_def += ")"
                    break
                line = fp.readline()
            
            print(function_def)
            function_doc = ""
            doc_found = False
            while line:
                print(line)
                if "\"\"\"" in line:
                    doc_found != doc_found
                elif doc_found:
                    function_doc += line

                line = fp.readline()

            print(function_doc)
                
#        print(function_def)
        if i == 2:
            break
        else:
            i += 1
    break
