# LearnBase

This package is an attempt to provide common interfaces and function definitions for Machine Learning packages in Julia

[![Build Status](https://travis-ci.org/Evizero/LearnBase.jl.svg?branch=master)](https://travis-ci.org/Evizero/LearnBase.jl)

Everything here is subject to change. The initial code here is just factored out code that i have currently in use

# Example

Base classes for common types of machine learning entities

```Julia
abstract AbstractLearner

fit(learner::AbstractLearner, args...; nargs...) = @_not_implemented

# Supervised Learner have targets
abstract AbstractSupervisedLearner

fit(learner::AbstractSupervisedLearner, formula::Formula, df::DataFrame, args...; nargs...) =
  fit(learner::AbstractSupervisedLearner, datasource(formula, df), args...; nargs...)

# Classifier are a special case of supervised learner
abstract AbstractClassifier <: AbstractSupervisedLearner

# Some classifier such as SVM need a specific encoding for their targets
abstract AbstractEncodedClassifier{E<:ClassEncoding} <: AbstractClassifier

# Using this information data sources such as DataFrames are encoded accordingly by default
fit{E<:ClassEncoding}(learner::AbstractEncodedClassifier{E}, ds::DataFrameLabeledDataSource, args...; nargs...) =
  fit(learner::AbstractSupervisedLearner, convert(EncodedInMemoryLabeledDataSource{E}, ds), args...; nargs...)
```

Datasources follow a common interface

```Julia
abstract DataSource

nobs(source::DataSource)
nvar(source::DataSource)
features(source::DataSource)
features(source::DataSource, offset::Int, length::Int) # for mini batches and online learning

# labeled sources

abstract LabeledDataSource <: DataSource
abstract InMemoryLabeledDataSource <: LabeledDataSource

groundtruth(source::LabeledDataSource) # the unencoded targets
targets(source::LabeledDataSource) # encoded targets
targets(source::LabeledDataSource, offset::Int, length::Int) # for mini batches and online learning
bias(source::LabeledDataSource) # some learner need the bias independent of the features
nclasses(source::LabeledDataSource)
labels(source::LabeledDataSource) # the unique labels (can be strings among things)
classdistribution(source::LabeledDataSource)
```
