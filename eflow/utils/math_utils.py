__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"


def get_unbalanced_threshold(target_amount,
                             max_binary_threshold=.65):
    max_unbalanced_class_threshold = (1 / target_amount) / ((1/2)/(max_binary_threshold-(1/2))) + (
                1 / target_amount)
    min_unbalanced_class_threshold = (1 - max_unbalanced_class_threshold) / (
                target_amount - 1)

    return max_unbalanced_class_threshold, min_unbalanced_class_threshold
