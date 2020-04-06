from setuptools import setup
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="gradient_descent",
    version="0.0.3",
    author="Daniel da Costa",
    author_email="daniel.pereiracosta@hotmail.com",
    description="Package for applying gradient descent optimization algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielDaCosta/optimization-algorithms",
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    zip_save=False
)
