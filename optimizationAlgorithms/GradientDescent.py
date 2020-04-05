import numpy as np
import matplotlib.pyplot as plt

from .ExceptionHandler import ConvergencePointsValueError


class GradientDescent():
    """Class containing differents methods for applying gradient descent method
    for single variable function

    Attributes:
        name (string): name of the optmizer
        f (function): function for optimization
        df (function): first derivation of the function
        x_t (float): starting variable for analysis
        learning_rate (float): learning rate
        tolerance (int): tolerance for the distance between two consecutive
        estimates in a subsequence that converges
        max_iterations (int): maximum number of iterations
        convergence_points (list): list to store the history of points during
        optimization n_iterations (int): number of iterations for convegence
    """
    def __init__(self, f, df, x_t, learning_rate=1e-3, tolerance=1e-6,
                 max_iterations=1000, n_history_points=1000):
        """Constructor

        Args:
            f (function): function for optimization
            df (function): first derivation of the function
            x_t (float): starting variable for analysis
            learning_rate (float, optional): learning rate
            tolerance (int, optional): tolerance for the distance between two
            consecutive estimates in a subsequence that converges
            max_iterations (int, optional): maximum number of iterations
            n_history_points (int, optional): total amount of history points
            to be saved during optization

        Returns:
            None
        """
        self.name = 'Gradient Descent'
        self.f = f
        self.df = df
        self.x_t = x_t
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_points = np.array([None]*n_history_points)
        self.n_iterations = 0

    def _update_parameter(self, x_t):
        """Computes the current update vector for GradientDescent

        Args:
            x_t (float): point for calculation

        Returns:
            (float): update amount
        """
        return self.learning_rate*self.df(x_t)

    def fit(self):
        """Gradient Descent for Optimization Algorithm

        Args:
            None

        Returns:
            (float) : local minimum measured by the algorithm
        """
        # Compute First Interation
        # Set new x_{t+1} = x_{t} - lambda*f'(x_{t})
        self.n_iterations = 1  # iteration step
        n_convergence_points = 0  # iteration step of list convergence_points

        x_t = self.x_t
        x_t_1 = x_t - self._update_parameter(x_t)

        self.convergence_points[n_convergence_points] = x_t

        while (np.abs(x_t_1 - x_t) > self.tolerance) and \
                (self.n_iterations <= self.max_iterations):
            try:
                # Update x_t
                x_t = x_t_1
                x_t_1 = x_t - self._update_parameter(x_t)

                # Stores some convergence points. Stores points that are
                # divisible by max_iterations*0.01.
                # Only some random points are stored in order not to create
                # a too big list and ending up
                # slowing down performance.
                if (self.n_iterations % (self.max_iterations*0.01) == 0):
                    n_convergence_points += 1
                    self.convergence_points[n_convergence_points] = x_t
                    if(n_convergence_points ==
                       (len(self.convergence_points) - 1)):
                        n_convergence_points = 0

                self.n_iterations += 1

            except OverflowError as err:
                print('Overflowed exception raised. The value of the function \
                    exploded. Try reducing the learning_rate')
                return err

        # If convergence_points list isn't completed, select only
        # the non-null values.
        self.convergence_points = \
            self.convergence_points[self.convergence_points is not None]\
            .astype(float)

        return x_t_1

    def get_n_iteration(self):
        """Get numbers of iterations required for optimization

        Args:
            None

        Returns:
            (int): number of iterations required
        """
        return self.n_iterations

    def plot_optimization(self, x1=None, x2=None, n_points=100):
        """Plotting function between interval (x1,x2)

        Args:
            x1 (float, optional): start of interval
            x2 (float, optional): end of interval
            n_points (int, optional): number of point to be plotted
            between x1 and x2

        Returns:
            None
        """
        try:
            if (len(self.convergence_points[self.convergence_points is None])):
                raise ConvergencePointsValueError('Empty x-axis. \
                    No Convergence Points')

            x1 = x1 or min(self.convergence_points)
            x2 = x2 or max(self.convergence_points)

            x_axis = np.linspace(x1, x2, n_points)
            y_x = self.f(x_axis)

            plt.plot(x_axis, y_x, color='blue')
            plt.ylabel('$f(x)$')
            plt.xlabel('$x$')
            plt.title(f'{self.name}')

            # Plot points on f(x)
            plt.plot(self.convergence_points, self.f(self.convergence_points),
                     'bo', color='red')
            plt.show()

        except ConvergencePointsValueError as err:
            print('Please, run the fit() method first: ', err)