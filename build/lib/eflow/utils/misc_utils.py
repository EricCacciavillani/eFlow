# General/Misc somethings that don't quite fit it the other utils
import itertools

def string_condtional(given_val,
                      full_condtional):
    """
    Simple string
    given_val:
        Numerical value to replace 'x'.

    full_condtional:
        Specified string conditional.
        Ex: x >= 0 and x <=100

    Returns/Descr:
        Returns back a boolean value of whether or not the conditional the
        given value passes the condition.

    Note:
        Currently only handles and conditionals (no 'or' statements yet)
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

