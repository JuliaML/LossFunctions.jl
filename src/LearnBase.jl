module LearnBase

using Reexport
@reexport using StatsBase
@reexport using MLBase
using ArrayViews
using DataFrames

import Base: show, shuffle!, convert
import StatsBase: fit, fit!, predict, nobs
import MLBase: labelencode, labeldecode, groupindices

export

    AbstractLearner,
    AbstractSupervisedLearner,
    AbstractClassifier,
    AbstractEncodedClassifier,

    ClassEncoding,
    BinaryClassEncoding,
    MultinomialClassEncoding,
    OneHotClassEncoding,

    nclasses,
    labels,
    classdistribution,

    DataSource,
    LabeledDataSource,
    InMemoryLabeledDataSource,
    EncodedInMemoryLabeledDataSource,
    DataFrameLabeledDataSource,

    nvar,
    features,
    targets,
    groundtruth,
    bias,

    datasource,
    encodeDataSource

include("common.jl")
include("classencoding.jl")
include("datasource.jl")
include("abstractlearner.jl")

end # module
