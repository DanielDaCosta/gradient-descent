from .GradientDescent import GradientDescent


class NAG(GradientDescent):
    """ NAG, Nesterov Accelerated Gradient, computes the function derivative
    base on the next position of the paramente. Looking ahead helps NAG in
    correcting its course quicker than Momentum based gradient descent.

    Attributes:
        f (function): function for optimization
        df (function): first derivation of the function
        x_t (float): starting variable for analysis
        learning_rate (float): learning rate
        tolerance (int): tolerance for the distance between two consecutive
        estimates in a subsequence that converges
        max_iterations (int): maximum number of iterations
        convergence_points (list): list to store the history of
        points during optimization
        n_iterations (int): number of iterations for convegence
        gamma (float): fraction of the past vector that contains an
        exponentially decaying average of past gradients
        u_t (float): Parameter for NAG optimization
        u_t_1 (float): Parameter for NAG optimization

    """

    def __init__(self, f, df, x_t, learning_rate=1e-3, tolerance=1e-6,
                 max_iterations=1000, n_history_points=1000, gamma=0.9):
        """Constructor
        Args:
            name (string): name of the optmizer
            f (function): function for optimization
            df (function): first derivation of the function
            x_t (float): starting variable for analysis
            learning_rate (float, optional): learning rate
            tolerance (int, optional): tolerance for the distance between two
            consecutive estimates in a subsequence that converges
            max_iterations (int, optional): maximum number of iterations
            n_history_points (int, optional): total amount of history points
            to be saved during optization
            gamma (float, optional): fraction of the past vector that contains
            an exponentially decaying average of past gradients

        Returns:
            None
        """
        GradientDescent.__init__(self, f, df, x_t, learning_rate, tolerance,
                                 max_iterations, n_history_points)
        self.name = 'NAG'
        self.gamma = gamma
        self.__u_t = 0
        self.__u_t_1 = 0

    def _update_parameter(self, x_t):
        """Computes the current update vector for Nesterov accelerated gradient

        Params:
            x_t (float): point for calculation

        Returns:
            (float): update amount
        """
        self.__u_t_1 = self.gamma*self.__u_t \
            + self.learning_rate*self.df(x_t - self.gamma*self.__u_t)
        self.__u_t = self.__u_t_1
        return self.__u_t_1
