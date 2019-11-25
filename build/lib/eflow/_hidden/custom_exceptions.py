__author__ = "Eric Cacciavillani"
__copyright__ = "Copyright 2019, eFlow"
__credits__ = ["Eric Cacciavillani"]
__license__ = "MIT"
__maintainer__ = "EricCacciavillani"
__email__ = "eric.cacciavillani@gmail.com"

class DefaultException(Exception):
    def __init__(self,
                 error_message=None):
        self.__error_message = error_message

    def __str__(self):
        if not self.__error_message:
            return "eflow has raised an exception without describing it!"
        else:
            return self.__error_message

class SnapshotMismatchError(DefaultException):
    def __init__(self,
                 error_message=None):
        super().__init__(error_message=error_message)

class UnsatisfiedRequirments(DefaultException):
    def __init__(self,
                 error_message=None):
        super().__init__(error_message=error_message)

class PipelineError(DefaultException):
    def __init__(self,
                 error_message=None):
        super().__init__(error_message=error_message)

class PipelineSegmentError(DefaultException):
    def __init__(self,
                 error_message=None):
        super().__init__(error_message=error_message)


class RequiresPredictionMethods(DefaultException):
    def __init__(self,
                 error_message=None):
        super().__init__(error_message=error_message)


class ProbasNotPossible(DefaultException):
    def __init__(self,
                 error_message=None):
        super().__init__(error_message=error_message)



# class UnknownPredictionType(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Couldn't assert whether type was a"
#                                        " float,int,list,or numpy array.")
#
# class PredictionTypeUnknown(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Can not logical assert what data type"
#                                        " is being returned from the model.")
#
# class ProbabliltyPrediction(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Can not logically assert what data type"
#                                        " is being returned from the model!")
#
# class ProbasNotPossible(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Check if the model probas/pred"
#                                        " function you init can return"
#                                        " probabilities.")
#
# class ThresholdLength(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Thersholds list must have the same"
#                                        " length as the targets!")
#
# class ThresholdType(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Thersholds must be a list or"
#                                        " numpy array!")
#
# class RequiresPredictionMethods(DefaultException):
#     def __init__(self):
#         super().__init__(error_message="Prediction methods must be passed as a dictionary.")
