from setuptools import setup, find_packages

setup(name='timeseries_toolkit',
      version='0.1dev',
      description='Helper library for time series forecasting',
      install_requires=[open('requirements.txt').read().splitlines()],
      packages=find_packages(),
      author='Alex Spanos',
      author_email='alexi.spanos@yahoo.com')