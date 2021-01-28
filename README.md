# gradient-descent

[![PyPI Latest Release](https://img.shields.io/pypi/v/gradient-descent.svg)](https://pypi.org/project/gradient-descent/)

gradient-descent is a package that contains different gradient-based algorithms, usually used to optimize Neural Networks and other machine learning models. The package contains the following algorithms:

- Gradients Descent
- Momentum
- RMSprop
- Nasterov accelerated gradient
- Adam

The package purpose is to facilitate the user experience when using optimization algorithms and to allow the user to have a better intuition about how these *black-boxes* algorithms work.

This is an open-source project, any feedback, improvement ideas, and contributors are welcome.

## Installation

**Dependencies**

- Python (>= 3.6)
- NumPy (>= 1.13.3)
- Matplotlib (>=3.2.1)

**User installation**

```
pip install gradient-descent
```

## Development

All contributors of all levels are welcome to help in any possible away. 

**Souce Code**

```
git clone https://github.com/DanielDaCosta/gradient-descent.git
```

**Tests**

```
pytest tests
```

## TO DO

The package is still on its early days and there are improvements to make. If you want to contribute to the project, you can start by addressing one of the items below:

- [ ] Build new optimization algorithms
- [ ] Extend its use for multivariable functions
- [ ] New ideas of functions for better usability
- [ ] Improve Documentation

# References & Acknowledgements

First of all I would like to thank Hammad Shaikh by his well documented and very well explained GitHub repository [Math of Machine Learning Course by Siraj](https://github.com/hammadshaikhha/Math-of-Machine-Learning-Course-by-Siraj/blob/master/Gradient%20Descent%20for%20Optimization/Gradient%20Descent%20for%20Optimization.ipynb)

I appreciate the help of the following contents and articles in the package development:

- [Optimizing Gradient Descent](https://ruder.io/optimizing-gradient-descent/) by Sebastian Ruder
- [Optimization Techniques for Gradient Descent](https://www.geeksforgeeks.org/optimization-techniques-for-gradient-descent/?ref=rp) by www.geeksforgeeks.org website
- [optimization_algos](https://github.com/idc9/optimization_algos) GitHub repository by Iain Carmichael
- [Deep Learning](http://www.deeplearningbook.org) by Begnio, Goodfellow and Courtville
