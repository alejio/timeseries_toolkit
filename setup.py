from setuptools import setup, find_packages

setup(name='timeseries_toolkit',
      version='0.1.1dev',
      description='Helper library for time series forecasting',
      install_requires=[open('requirements.txt').read().splitlines()],
      packages=find_packages(),
      author='Alex Spanos',
      url = 'https://github.com/alejio/timeseries_toolkit',
      author_email='alexi.spanos@yahoo.com',
      classifiers = [
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
      ]
)