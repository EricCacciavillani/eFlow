import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os
import copy
import six


def enum(**enums):
    return type('Enum', (), enums)

def check_create_dir_structure(directory_pth,
                               sub_dir):
    """
        Checks/Creates required directory structures inside
        the parent directory figures.
    """

    for dir in sub_dir.split("/"):
        directory_pth += "/" + dir
        if not os.path.exists(directory_pth):
            os.makedirs(directory_pth)

    return directory_pth


def create_plt_png(directory_pth,
                   sub_dir,
                   filename):
    """
        Saves the plt based image in the correct directory.
    """

    # Ensure directory structure is init correctly
    abs_path = check_create_dir_structure(directory_pth,
                                          sub_dir)

    # Ensure file ext is on the file.
    if filename[-4:] != ".png":
        filename += ".png"

    fig = plt.figure(1)
    fig.savefig(abs_path + "/" + filename, bbox_inches='tight')


def df_to_image(df,
                directory_pth,
                sub_dir,
                filename,
                col_width=5.0,
                row_height=0.625,
                font_size=14,
                header_color='#40466e',
                row_colors=['#f1f1f2', 'w'],
                edge_color='w',
                bbox=[0, 0, 1, 1],
                header_columns=0,
                ax=None,
                show_index=False,
                index_color="#add8e6",
                format_float_pos=None,
                **kwargs):
    df = copy.deepcopy(df)

    if format_float_pos and format_float_pos > 1:
        float_format = '{:,.' + str(2) + 'f}'
        for col_feature in set(df.select_dtypes(include=["float"]).columns):
            df[col_feature] = df[col_feature].map(float_format.format)

    if ax is None:
        size = (np.array(df.shape[::-1]) + np.array([0, 1])) * np.array(
            [col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    if show_index:
        df.reset_index(inplace=True)

    mpl_table = ax.table(cellText=df.values, bbox=bbox,
                         colLabels=df.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            if index_color and show_index and k[1] == 0:
                cell.set_facecolor(index_color)
            else:
                cell.set_facecolor(row_colors[k[0] % len(row_colors)])

    create_plt_png(directory_pth,
                   sub_dir,
                   filename)
    plt.close()


def string_condtional(given_val,
                      full_condtional):
    """
    Simple string
    given_val:
        Numerical value to replace 'x'
    full_condtional:
        Specified string conditional.
        Ex: x >= 0 and x <=100

    Returns/Descr:
        Returns back a boolean value of whether or not the conditional the
        given value passes the condition.
    """
    condtional_returns = []
    operators = [i for i in full_condtional.split(" ")
                 if i == "or" or i == "and"]

    all_condtionals = list(
        itertools.chain(
            *[i.split("or")
              for i in full_condtional.split("and")]))
    for condtional_line in all_condtionals:
        condtional_line = condtional_line.replace(" ", "")
        if condtional_line[0] == 'x':
            condtional_line = condtional_line.replace('x', '')

            condtional = ''.join(
                [i for i in condtional_line if
                 not (i.isdigit() or i == '.')])
            compare_val = float(condtional_line.replace(condtional, ''))
            if condtional == "<=":
                condtional_returns.append(given_val <= compare_val)

            elif condtional == "<":
                condtional_returns.append(given_val < compare_val)

            elif condtional == ">=":
                condtional_returns.append(given_val >= compare_val)

            elif condtional == ">":
                condtional_returns.append(given_val > compare_val)

            elif condtional == "==":
                condtional_returns.append(given_val == compare_val)

            elif condtional == "!=":
                condtional_returns.append(given_val != compare_val)
            else:
                print("ERROR")
                return False
        else:
            print("ERROR")
            return False

    if not len(operators):
        return condtional_returns[0]
    else:
        i = 0
        final_return = None

        for op in operators:
            print(condtional_returns)
            if op == "and":

                if final_return is None:
                    final_return = condtional_returns[i] and \
                                   condtional_returns[i + 1]
                    i += 2
                else:
                    final_return = final_return and condtional_returns[i]
                    i += 1

            else:
                if final_return is None:
                    final_return = condtional_returns[i] or \
                                   condtional_returns[i + 1]
                    i += 2
                else:
                    final_return = final_return or condtional_returns[i]
                    i += 1

        return final_return