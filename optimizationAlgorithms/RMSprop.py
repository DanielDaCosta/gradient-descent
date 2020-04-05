from .GradientDescent import GradientDescent
import numpy as np


class RMSprop(GradientDescent):
    """This algorithm follows the inuition about applying
    an exponentially weighted average method to
    the second moment of the gradients (df2)

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
        beta_2 (float): fraction of the past vector that contains an
        exponentially decaying average of squared gradients
        s_t (float): Parameter for optimization of RMSprop
        s_t_1 (float): Parameter for optimization of RMSprop
    """

    def __init__(self, f, df, x_t, learning_rate=1e-3, tolerance=1e-6,
                 max_iterations=1000, n_history_points=1000, beta_2=0.9):
        """Constructor
        Args:
            f (function): function for optimization
            df (function): first derivation of the function
            x_t (float): starting variable for analysis
            learning_rate (float, optional): learning rate
            tolerance (int, optional): tolerance for the distance between
            two consecutive estimates in a subsequence that converges
            max_iterations (int, optional): maximum number of iterations
            n_history_points (int, optional): total amount of history points
            to be saved during optization
            beta_2 (float, optional): fraction of the past vector that
            contains an exponentially decaying average of
            past squared gradients.

        Returns:
            None
        """
        GradientDescent.__init__(self, f, df, x_t, learning_rate, tolerance,
                                 max_iterations, n_history_points)
        self.name = 'RMSprop'
        self.beta_2 = beta_2
        self.__s_t = 0
        self.__s_t_1 = 0

    def _update_parameter(self, x_t):
        """Computes the current update vector for RMSprop

        Params:
            x_t (float): point for calculation

        Returns:
            (float): update amount

        """
        epsilon = 1e-8

        self.__s_t_1 = self.beta_2*self.__s_t + \
            (1 - self.beta_2)*self.df(x_t)**2
        self.__s_t = self.__s_t_1

        return self.learning_rate*self.df(x_t)/(np.sqrt(self.__s_t_1)+epsilon)
