import unittest
import numpy as np

from gradient_descent.RMSprop import RMSprop


class TestRMSpropClass(unittest.TestCase):

    def setUp(self):
        """Setting up requirements for test
        Params:
            None
        Returns:
            None
        """
        def f(x):
            """Apply function to point x

            Args:
                x (float): point on x-axis

            Returns:
                (float): f(x)
            """
            return 4*x**2

        def df(x):
            """Apply function gradient to point x

            Args:
                x (float): point on x-axis

            Returns:
                (float): df(x)
            """
            return 8*x

        self.rmsprop = RMSprop(f, df, x_t=10, learning_rate=0.1,
                               max_iterations=1000, tolerance=1e-6,
                               n_history_points=1000, beta_2=0.9)

    def test_initizialization(self):
        """Testing Attributes initialization

        Args:
            None

        Returns:
            None
        """
        self.assertEqual(self.rmsprop.x_t, 10,
                         'incorrect initial value of x_t')
        self.assertEqual(self.rmsprop.learning_rate, 0.1,
                         'incorrect value of learning_rate')
        self.assertEqual(self.rmsprop.max_iterations, 1000,
                         'incorrect value of max_iterations')
        self.assertEqual(self.rmsprop.tolerance, 1e-6,
                         'incorrect value of tolerance')
        self.assertEqual(self.rmsprop.n_iterations, 0,
                         'incorrect value of n_iterations')
        np.testing.assert_array_equal(self.rmsprop.convergence_points,
                                      np.array([None]*1000),
                                      'incorrect inialization of array \
                                          convergence_points')
        self.assertEqual(self.rmsprop.beta_2, 0.9,
                         'incorrect initilialization of beta_1')
        self.assertEqual(self.rmsprop._RMSprop__s_t, 0,
                         'incorrect initialization of s_t')
        self.assertEqual(self.rmsprop._RMSprop__s_t_1, 0,
                         'incorrect initialization of s_t_1')

    def test_update_parameter(self):
        """Testing _update_parameter method

        Args:
            None

        Returns:
            None
        """
        s_t = self.rmsprop._RMSprop__s_t
        epsilon = 1e-8
        self.assertAlmostEqual(self.rmsprop._update_parameter(10),
                               self.rmsprop.learning_rate
                               * self.rmsprop.df(10)
                               / (np.sqrt(self.rmsprop.beta_2*s_t
                                  + (1-self.rmsprop.beta_2)
                                  * self.rmsprop.df(10)**2) + epsilon),
                               'incorrect return of _update_parameter')

        s_t = self.rmsprop._RMSprop__s_t
        self.assertAlmostEqual(self.rmsprop._update_parameter(3),
                               self.rmsprop.learning_rate
                               * self.rmsprop.df(3)
                               / (np.sqrt(self.rmsprop.beta_2*s_t
                                          + (1-self.rmsprop.beta_2)
                                          * self.rmsprop.df(3)**2) + epsilon),
                               'incorrect return of _update_parameter')

    def test_optimization(self):
        """Test the optimization algorithm

        Args:
            None
        Returns:
            None
        """
        self.assertLessEqual(self.rmsprop.fit(), 1e-5,
                             'Failed to converge to zero for the function: \
                                 4x**2')
        self.assertGreaterEqual(self.rmsprop.n_iterations, 1,
                                "n_iterations wasn't properly updated")


if __name__ == '__main__':
    unittest.main()
