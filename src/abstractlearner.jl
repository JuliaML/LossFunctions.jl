abstract AbstractLearner

fit(learner::AbstractLearner, args...; nargs...) = @_not_implemented

# ==========================================================================
# Supervised Learner have targets

abstract AbstractSupervisedLearner

fit(learner::AbstractSupervisedLearner, formula::Formula, df::DataFrame, args...; nargs...) =
  fit(learner::AbstractSupervisedLearner, datasource(formula, df), args...; nargs...)

# ==========================================================================
# Classifier are a special case of supervised learner

abstract AbstractClassifier <: AbstractSupervisedLearner

# ==========================================================================
# Some classifier such as SVM need a specific encoding for their targets

abstract AbstractEncodedClassifier{E<:ClassEncoding} <: AbstractClassifier

fit{E<:ClassEncoding}(learner::AbstractEncodedClassifier{E}, ds::DataFrameLabeledDataSource, args...; nargs...) =
  fit(learner::AbstractSupervisedLearner, convert(EncodedInMemoryLabeledDataSource{E}, ds), args...; nargs...)

