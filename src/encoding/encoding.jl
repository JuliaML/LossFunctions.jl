
include("classes.jl")
include("model.jl")

typealias OneHotClassEncoding OneOfKClassEncoding

export
    ClassEncoding,
        SignedClassEncoding,
        BinaryClassEncoding,
        MultinomialClassEncoding,
        OneHotClassEncoding,
    ClassEncodings,

    nclasses,
    labels,
    classdistribution,

    EncodedStatisticalModel,
    EncodedRegressionModel

@autocomplete ClassEncodings export
    ZeroOneClassEncoding,
    SignedClassEncoding,
    MultivalueClassEncoding,
    OneOfKClassEncoding,
    OneHotClassEncoding
