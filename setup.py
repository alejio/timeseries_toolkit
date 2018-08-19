from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='timeseries_toolkit',
      version='0.1.1dev',
      description='Helper library for time series forecasting',
      long_description = long_description,
      long_description_content_type="text/markdown",
      install_requires=[open('requirements.txt').read().splitlines()],
      packages=find_packages(),
      author='Alex Spanos',
      author_email='alexi.spanos@yahoo.com',
      url='https://github.com/alejio/timeseries_toolkit',
      classifiers=[
          "Programming Language :: Python :: 3",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent",
      ]
      )
