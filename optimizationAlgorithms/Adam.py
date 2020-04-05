from .GradientDescent import GradientDescent
import numpy as np


class Adam(GradientDescent):
    """Adam optimization algorithm incorporates the momentum method and RMSprop,
    along with bias correction.

    Attributes:
        name (string): name of the optmizer
        f (function): function for optimization
        df (function): first derivation of the function
        x_t (float): starting variable for analysis
        learning_rate (float): learning rate
        tolerance (int): tolerance for the distance between two consecutive
        estimates in a subsequence that converges
        max_iterations (int): maximum number of iterations
        convergence_points (list): list to store the history of points
        during optimization
        n_iterations (int): number of iterations for convegence
        beta_1 (float): fraction of the update vector of the past time
        step to the current update
        beta_2 (float): fraction of the past vector that contains an
        exponentially decaying average of squared gradients
        m_t (float): Parameter for storing exponentially decaying average of
        past grandients
        m_t_1 (float): Parameter for storing exponentially decaying average of
        past grandients
        v_t (float): Parameter for storing exponentially decaying average of
        past squared grandients
        v_t_1 (float): Parameter for storing exponentially decaying average of
        past squared grandients
    """
    def __init__(self, f, df, x_t, learning_rate=1e-3, tolerance=1e-6,
                 max_iterations=1000, n_history_points=1000, beta_1=0.9,
                 beta_2=0.999):
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
            beta_1 (float, optional): fraction of the past vector that
            contains an exponentially decaying average of past gradients
            beta_2 (float, optional): fraction of the past vector that
            contains an exponentially decaying average of
            past squared gradients

        Returns:
            None
        """
        GradientDescent.__init__(self, f, df, x_t, learning_rate, tolerance, 
                                 max_iterations, n_history_points)
        self.name = 'Adam Optimizer'
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.__m_t = 0
        self.__m_t_1 = 0
        self.__v_t = 0
        self.__v_t_1 = 0

    def _update_parameter(self, x_t):
        """Computes the current update vector for Adam Optimizer

        Params:
            x_t (float): point for calculation

        Returns:
            (float): update amount
        """

        # Exponentially decaying average of past gradient m_t
        self.__m_t_1 = self.beta_1*self.__m_t + (1 - self.beta_1)*self.df(x_t)
        self.__m_t = self.__m_t_1

        # Exponentially decaying average of past squared gradients v_t
        self.__v_t_1 = self.beta_2*self.__v_t + (1-self.beta_2)*self.df(x_t)**2
        self.__v_t = self.__v_t_1

        # Adam includes bias correction to the estimates of both the
        # first-moments (the momentum term) and the second order moments
        # to account for their initialization at the origin

        m_hat_t = self.__m_t_1/(1 - self.beta_1**self.n_iterations)
        v_hat_t = self.__v_t_1/(1 - self.beta_2**self.n_iterations)

        epsilon = 1e-8  # handling division by zero

        return self.learning_rate*m_hat_t/(np.sqrt(v_hat_t) + epsilon)
