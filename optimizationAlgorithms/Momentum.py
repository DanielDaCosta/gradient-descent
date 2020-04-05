from .GradientDescent import GradientDescent


class Momentum(GradientDescent):
    """This method is used to accelerate the gradient descent
    algorithm by taking into consideration the exponentially weighted average
    of the gradients.

    Attributes:
        name (string): name of the optmizer
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
        beta_1 (float): fraction of the update vector of the past time step to
        the current update
        v_t (float): Parameter for Momentum optimization
        v_t_1 (float): Parameter for Momentum optimization
    """

    def __init__(self, f, df, x_t, learning_rate=1e-3, tolerance=1e-6,
                 max_iterations=1000, n_history_points=1000, beta_1=0.9):
        """Constructor
        Args:
            f (function): function for optimization
            df (function): first derivation of the function
            x_t (float): starting variable for analysis
            learning_rate (float, optional): learning rate
            tolerance (int, optional): tolerance for the distance
            between two consecutive estimates in a subsequence that converges
            max_iterations (int, optional): maximum number of iterations
            n_history_points (int, optional): total amount of history points
            to be saved during optization
            beta_1 (float, optional): fraction of the update vector of the
            past time step to the current update

        Returns:
            None
        """
        GradientDescent.__init__(self, f, df, x_t, learning_rate, tolerance,
                                 max_iterations, n_history_points)
        self.name = 'Momentum'
        self.beta_1 = beta_1
        self.__v_t = 0
        self.__v_t_1 = 0

    def _update_parameter(self, x_t):
        """Computes the current update vector for Momentum

        Params:
            x_t (float): point for calculation

        Returns:
            (float): update amount
        """
        self.__v_t_1 = self.beta_1*self.__v_t + (1 - self.beta_1)*self.df(x_t)
        self.__v_t = self.__v_t_1

        return self.learning_rate*self.__v_t_1
