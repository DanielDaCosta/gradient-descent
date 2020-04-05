class ConvergencePointsValueError(Exception):
    """Exception class for handling ConvergencePoints error: when the
    optimization algorithm hasn't been ran, and its results are accesed
    by mistake

    Attributes:
        message (str): error message
    """

    def __init__(self, message):

        self.message = message

    def __str__(self):
        return repr(self.message)