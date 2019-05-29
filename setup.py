from setuptools import setup

setup(
    name='constrained_cem_mpc',
    version='0.1.0dev',
    author='Oscar Key',
    author_email='oscar.t.key@gmail.com',
    packages=['constrained_cem_mpc', 'constrained_cem_mpc.test'],
    install_requires=['torch', 'numpy', 'matplotlib', 'pytest', 'pytest-mock', 'gym', 'polytope']
)