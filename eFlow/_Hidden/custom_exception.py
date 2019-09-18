from eflow._hidden.Objects.DefaultException import *

class UnknownPredictionType(DefaultException):
    def __init__(self):
        super().__init__(error_message="Couldn't assert whether type was a"
                                       " float,int,list,or numpy array.")

class PredictionTypeUnknown(DefaultException):
    def __init__(self):
        super().__init__(error_message="Can not logical assert what data type"
                                       " is being returned from the model.")

class ProbabliltyPrediction(DefaultException):
    def __init__(self):
        super().__init__(error_message="Can not logically assert what data type"
                                       " is being returned from the model!")

class ProbasNotPossible(DefaultException):
    def __init__(self):
        super().__init__(error_message="Check if the model probas/pred"
                                       " function you init can return"
                                       " probabilities.")

class ThresholdLength(DefaultException):
    def __init__(self):
        super().__init__(error_message="Thersholds list must have the same"
                                       " length as the targets!")

class ThresholdType(DefaultException):
    def __init__(self):
        super().__init__(error_message="Thersholds must be a list or"
                                       " numpy array!")

class RequiresPredictionMethods(DefaultException):
    def __init__(self):
        super().__init__(error_message="Prediction methods must be passed as a dictionary.")
