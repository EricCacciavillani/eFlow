class DefaultException(Exception):
    def __init__(self,
                 error_message=None):
        self.__error_message = error_message

    def __str__(self):
        if not self.__error_message:
            return "eflow has raised an undeclared exception"
        else:
            return self.__error_message