import unittest
import numpy as np


from gradient_descent.Adam import Adam


class TestAdamClass(unittest.TestCase):

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

        self.adam = Adam(f, df, x_t=10, learning_rate=0.1,
                         max_iterations=1000, tolerance=1e-6,
                         n_history_points=1000, beta_1=0.9, beta_2=0.999)

    def test_initizialization(self):
        """Testing Attributes initialization

        Args:
            None

        Returns:
            None
        """
        self.assertEqual(self.adam.x_t, 10, 'incorrect initial value of x_t')
        self.assertEqual(self.adam.learning_rate, 0.1,
                         'incorrect value of learning_rate')
        self.assertEqual(self.adam.max_iterations, 1000,
                         'incorrect value of max_iterations')
        self.assertEqual(self.adam.tolerance, 1e-6,
                         'incorrect value of tolerance')
        self.assertEqual(self.adam.n_iterations, 0,
                         'incorrect value of n_iterations')
        np.testing.assert_array_equal(self.adam.convergence_points,
                                      np.array([None]*1000),
                                      'incorrect inialization of array\
                                        convergence_points')
        self.assertEqual(self.adam.beta_1, 0.9,
                         'incorrect initilialization of beta_1')
        self.assertEqual(self.adam.beta_2, 0.999,
                         'incorrect initilialization of beta_2')
        self.assertEqual(self.adam._Adam__m_t, 0,
                         'incorrect initialization of m_t')
        self.assertEqual(self.adam._Adam__m_t_1, 0,
                         'incorrect initialization of m_t_1')
        self.assertEqual(self.adam._Adam__v_t, 0,
                         'incorrect initialization of v_t')
        self.assertEqual(self.adam._Adam__v_t_1, 0,
                         'incorrect initialization of v_t_1')

    def test_update_parameter(self):
        """Testing _update_parameter method

        Args:
            None

        Returns:
            None
        """

        # Testing for x_t = 10
        epsilon = 1e-8
        test_x_t = 10
        self.adam.n_iterations = 1
        m_t = self.adam._Adam__m_t
        v_t = self.adam._Adam__v_t

        m_t_1 = self.adam.beta_1*m_t + (1 - self.adam.beta_1) \
            * self.adam.df(test_x_t)
        v_t_1 = self.adam.beta_2*v_t + (1 - self.adam.beta_2) \
            * self.adam.df(test_x_t)**2

        m_hat_t = m_t_1/(1 - self.adam.beta_1**self.adam.n_iterations)
        v_hat_t = v_t_1/(1 - self.adam.beta_2**self.adam.n_iterations)

        self.assertAlmostEqual(self.adam._update_parameter(test_x_t),
                               self.adam.learning_rate
                               * m_hat_t/(np.sqrt(v_hat_t) + epsilon),
                               'incorect return of _update parameter')

        # Testing for x_t = 3
        epsilon = 1e-8
        test_x_t = 3
        m_t = self.adam._Adam__m_t
        v_t = self.adam._Adam__v_t

        m_t_1 = self.adam.beta_1*m_t + (1 - self.adam.beta_1) \
            * self.adam.df(test_x_t)
        v_t_1 = self.adam.beta_2*v_t + (1 - self.adam.beta_2) \
            * self.adam.df(test_x_t)**2

        m_hat_t = m_t_1/(1 - self.adam.beta_1**self.adam.n_iterations)
        v_hat_t = v_t_1/(1 - self.adam.beta_2**self.adam.n_iterations)

        self.assertAlmostEqual(self.adam._update_parameter(test_x_t),
                               self.adam.learning_rate *
                               m_hat_t/(np.sqrt(v_hat_t) + epsilon),
                               'incorect return of _update parameter')

    def test_optimization(self):
        """Test the optimization algorithm

        Args:
            None
        Returns:
            None
        """
        self.assertLessEqual(self.adam.fit(), 1e-4,
                             'Failed to converge to zero for the function: \
                                 4x**2')
        self.assertGreaterEqual(self.adam.n_iterations, 1,
                                "n_iterations wasn't properly updated")


if __name__ == '__main__':
    unittest.main()
