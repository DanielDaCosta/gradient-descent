import unittest
import numpy as np

from gradient_descent.GradientDescent import GradientDescent


class TestGradientDescentClass(unittest.TestCase):

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

        self.gradient_descent = GradientDescent(f, df, x_t=10,
                                                learning_rate=0.1,
                                                max_iterations=1000,
                                                tolerance=1e-6,
                                                n_history_points=1000)

    def test_initizialization(self):
        """Testing Attributes initialization

        Args:
            None

        Returns:
            None
        """
        self.assertEqual(self.gradient_descent.x_t, 10,
                         'incorrect initial value of x_t')
        self.assertEqual(self.gradient_descent.learning_rate, 0.1,
                         'incorrect value of learning_rate')
        self.assertEqual(self.gradient_descent.max_iterations, 1000,
                         'incorrect value of max_iterations')
        self.assertEqual(self.gradient_descent.tolerance, 1e-6,
                         'incorrect value of tolerance')
        self.assertEqual(self.gradient_descent.n_iterations, 0,
                         'incorrect value of n_iterations')
        np.testing.assert_array_equal(self.gradient_descent.convergence_points,
                                      np.array([None]*1000),
                                      'incorrect inialization of array \
                                          convergence_points')

    def test_update_parameter(self):
        """Testing _update_parameter method

        Args:
            None

        Returns:
            None
        """
        self.assertAlmostEqual(self.gradient_descent._update_parameter(10),
                               0.1*self.gradient_descent.df(10),
                               'incorrect return of _update_parameter')
        self.assertAlmostEqual(self.gradient_descent._update_parameter(3),
                               0.1*self.gradient_descent.df(3),
                               'incorrect return of _update_parameter')

    def test_optimization(self):
        """Test the optimization algorithm

        Args:
            None
        Returns:
            None
        """
        self.assertLessEqual(self.gradient_descent.fit(), 1e-6,
                             'Failed to converge to zero for the function: \
                                 4x**2')
        self.assertGreaterEqual(self.gradient_descent.n_iterations, 1,
                                "n_iterations wasn't properly updated")


if __name__ == '__main__':
    unittest.main()
    