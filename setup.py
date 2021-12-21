import os
from setuptools import find_packages, setup

os.chdir(os.path.dirname(os.path.abspath(__file__)))

setup(
    name='marl_coop',
    packages=find_packages(),
    version='0.0.1',
    description='Implementation of a deep reinforcement learning agents that solve a cooperation tennis game.',
    author='Pierre Massey',
    license='',
    url="https://github.com/PierreMsy/DRL_cooperation.git",
    include_package_data=True
)