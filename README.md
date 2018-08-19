# timeseries_toolkit

**Author**: Alex Spanos

**Date**: 09/08/18

* Latest version: `0.1.5dev`


## Overview

This repository contains helper code for time series forecasting
problems.

The aim for the package is to provide a unified interface for tackling
time series forecasting problems, by allowing a quick and easy performance
comparison for standard methodologies such as the
[naive method](https://en.wikipedia.org/wiki/Forecasting#Na%C3%AFve_approach),
[exponential smoothing](https://en.wikipedia.org/wiki/Exponential_smoothing),
[ARIMA](https://en.wikipedia.org/wiki/Autoregressive_integrated_moving_average) and standard machine
 learning.



## Installation

### Dependencies

timeseries_toolkit requires

- Python (3.6)
- Numpy
- Pandas
- Scipy

### User installation

`pip install -U timeseries-toolkit`

**Note**: if installation fails on MacOSX because of `pip` failing to
build wheels for `psutil`, the reason is most likely due to a `gcc`
incompatibility. Doing `conda install gcc` before trying to install
`pip install -U timeseries-toolkit` should fix the problem.

### Changelog