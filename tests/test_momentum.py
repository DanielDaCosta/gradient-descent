import unittest
import numpy as np

from gradient_descent.Momentum import Momentum


class TestMomentumClass(unittest.TestCase):

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

        self.momentum = Momentum(f, df, x_t=10, learning_rate=0.1,
                                 max_iterations=1000, tolerance=1e-6,
                                 n_history_points=1000, beta_1=0.9)

    def test_initizialization(self):
        """Testing Attributes initialization

        Args:
            None

        Returns:
            None
        """
        self.assertEqual(self.momentum.x_t, 10,
                         'incorrect initial value of x_t')
        self.assertEqual(self.momentum.learning_rate, 0.1,
                         'incorrect value of learning_rate')
        self.assertEqual(self.momentum.max_iterations, 1000,
                         'incorrect value of max_iterations')
        self.assertEqual(self.momentum.tolerance, 1e-6,
                         'incorrect value of tolerance')
        self.assertEqual(self.momentum.n_iterations, 0,
                         'incorrect value of n_iterations')
        np.testing.assert_array_equal(self.momentum.convergence_points,
                                      np.array([None]*1000),
                                      'incorrect inialization of array \
                                          convergence_points')
        self.assertEqual(self.momentum.beta_1, 0.9,
                         'incorrect initilialization of beta_1')
        self.assertEqual(self.momentum._Momentum__v_t, 0,
                         'incorrect initialization of v_t')
        self.assertEqual(self.momentum._Momentum__v_t_1, 0,
                         'incorrect initialization of v_t_1')

    def test_update_parameter(self):
        """Testing _update_parameter method

        Args:
            None

        Returns:
            None
        """
        v_t = self.momentum._Momentum__v_t
        self.assertAlmostEqual(self.momentum._update_parameter(10),
                               self.momentum.learning_rate
                               * (self.momentum.beta_1*v_t
                               + (1-self.momentum.beta_1)
                               * self.momentum.df(10)),
                               'incorrect return of _update_parameter')

        v_t = self.momentum._Momentum__v_t
        self.assertAlmostEqual(self.momentum._update_parameter(3),
                               self.momentum.learning_rate
                               * (self.momentum.beta_1*v_t
                               + (1-self.momentum.beta_1)*self.momentum.df(3)),
                               'incorrect return of _update_parameter')

    def test_optimization(self):
        """Test the optimization algorithm

        Args:
            None
        Returns:
            None
        """
        self.assertLessEqual(self.momentum.fit(), 1e-4,
                             'Failed to converge to zero for the function: \
                                 4x**2')
        self.assertGreaterEqual(self.momentum.n_iterations, 1,
                                "n_iterations wasn't properly updated")


if __name__ == '__main__':
    unittest.main()
