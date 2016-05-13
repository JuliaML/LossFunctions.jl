# LearnBase

## WORK IN PROGRESS

[![Project Status: WIP - Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](http://www.repostatus.org/badges/latest/wip.svg)](http://www.repostatus.org/#wip)

This package is an attempt to provide common interfaces and function definitions for Machine Learning packages in Julia

[![Build Status](https://travis-ci.org/Evizero/LearnBase.jl.svg?branch=master)](https://travis-ci.org/Evizero/LearnBase.jl)

Everything here is subject to change. The initial code here is just factored out code that I have currently in use.

# Example

Common class encodings for machine learning algorithms that need numeric target vectors

```Julia
ZeroOneClassEncoding,
SignedClassEncoding,
MultivalueClassEncoding,
OneOfKClassEncoding
```

Abstract types for convenience. Extending from them instead of the base types
will automatically take care of non-numeric target vectors using a decorator
Their behavious is very similar to `DataFrameModels` do.

```Julia
abstract EncodedStatisticalModel{E<:ClassEncoding} <: StatisticalModel
abstract EncodedRegressionModel{E<:ClassEncoding} <: RegressionModel
```
