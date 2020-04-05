import unittest
import numpy as np

from gradient_descent.NAG import NAG


class TestNAGClass(unittest.TestCase):

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

        self.nag = NAG(f, df, x_t=10, learning_rate=0.1, max_iterations=1000,
                       tolerance=1e-6, n_history_points=1000, gamma=0.9)

    def test_initizialization(self):
        """Testing Attributes initialization

        Args:
            None

        Returns:
            None
        """
        self.assertEqual(self.nag.x_t, 10,
                         'incorrect initial value of x_t')
        self.assertEqual(self.nag.learning_rate, 0.1,
                         'incorrect value of learning_rate')
        self.assertEqual(self.nag.max_iterations, 1000,
                         'incorrect value of max_iterations')
        self.assertEqual(self.nag.tolerance, 1e-6,
                         'incorrect value of tolerance')
        self.assertEqual(self.nag.n_iterations, 0,
                         'incorrect value of n_iterations')
        np.testing.assert_array_equal(self.nag.convergence_points,
                                      np.array([None]*1000),
                                      'incorrect inialization of array \
                                          convergence_points')
        self.assertEqual(self.nag.gamma, 0.9,
                         'incorrect initilialization of gamma')
        self.assertEqual(self.nag._NAG__u_t, 0,
                         'incorrect initialization of u_t')
        self.assertEqual(self.nag._NAG__u_t_1, 0,
                         'incorrect initialization of u_t_1')

    def test_update_parameter(self):
        """Testing _update_parameter method

        Args:
            None

        Returns:
            None
        """
        u_t = self.nag._NAG__u_t
        self.assertAlmostEqual(self.nag._update_parameter(10),
                               self.nag.gamma*u_t + self.nag.learning_rate
                               * self.nag.df(10 - self.nag.gamma*u_t),
                               'incorrect return of _update_parameter')
        u_t = self.nag._NAG__u_t
        self.assertAlmostEqual(self.nag._update_parameter(3),
                               self.nag.gamma*u_t + self.nag.learning_rate
                               * self.nag.df(3 - self.nag.gamma*u_t),
                               'incorrect return of _update_parameter')

    def test_optimization(self):
        """Test the optimization algorithm

        Args:
            None
        Returns:
            None
        """
        self.assertLessEqual(self.nag.fit(), 1e-5,
                             'Failed to converge to zero for the function: \
                                 4x**2')
        self.assertGreaterEqual(self.nag.n_iterations, 1,
                                "n_iterations wasn't properly updated")


if __name__ == '__main__':
    unittest.main()
