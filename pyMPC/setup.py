from setuptools import setup, find_packages
import sys

setup(
    name='Vehicle_MPC',
    version= '0.1',
    py_modules=['valet_parking'],
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'cvxpy',
    ],
    description="A simple MPC controller for vehicle agent with bycycle model",
    author="Wang Pengyu",
)
