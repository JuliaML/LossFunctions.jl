"""
Baseclass for all aggregation modes.
"""
abstract type AggregateMode end

"""
    module AggMode

Types for aggregation of multiple observations.

- `AggMode.Sum()`
- `AggMode.Mean()`
- `AggMode.WeightedSum(weights)`
- `AggMode.WeightedMean(weights)`
"""
module AggMode
    using ..LossFunctions: AggregateMode

    """
        AggMode.Sum()

    Causes the method to return the unweighted sum of the
    elements instead of the individual elements. Can be used in
    combination with `ObsDim`, in which case a vector will be
    returned containing the sum for each observation (useful
    mainly for multivariable regression).
    """
    struct Sum <: AggregateMode end

    """
        AggMode.Mean()

    Causes the method to return the unweighted mean of the
    elements instead of the individual elements. Can be used in
    combination with `ObsDim`, in which case a vector will be
    returned containing the mean for each observation (useful
    mainly for multivariable regression).
    """
    struct Mean <: AggregateMode end

    """
        AggMode.WeightedSum(weights; [normalize = false])

    Causes the method to return the weighted sum of all
    observations. The variable `weights` has to be a vector of
    the same length as the number of observations.
    If `normalize = true`, the values of the weight vector will
    be normalized in such as way that they sum to one.

    # Arguments

    - `weights::AbstractVector`: Vector of weight values that
      can be used to give certain observations a stronger
      influence on the sum.

    - `normalize::Bool`: Boolean that specifies if the weight
      vector should be transformed in such a way that it sums to
      one (i.e. normalized). This will not mutate the weight
      vector but instead happen on the fly during the
      accumulation.

      Defaults to `false`. Setting it to `true` only really
      makes sense in multivalue-regression, otherwise the result
      will be the same as for [`WeightedMean`](@ref).
    """
    struct WeightedSum{W<:AbstractVector} <: AggregateMode
        weights::W
        normalize::Bool
    end
    WeightedSum(weights::AbstractVector; normalize::Bool = false) = WeightedSum(weights, normalize)

    """
        AggMode.WeightedMean(weights; [normalize = true])
    
    Causes the method to return the weighted mean of all
    observations. The variable `weights` has to be a vector of
    the same length as the number of observations.
    If `normalize = true`, the values of the weight vector will
    be normalized in such as way that they sum to one.

    # Arguments

    - `weights::AbstractVector`: Vector of weight values that can
      be used to give certain observations a stronger influence
      on the mean.

    - `normalize::Bool`: Boolean that specifies if the weight
      vector should be transformed in such a way that it sums to
      one (i.e. normalized). This will not mutate the weight
      vector but instead happen on the fly during the
      accumulation.

      Defaults to `true`. Setting it to `false` only really makes
      sense in multivalue-regression, otherwise the result will
      be the same as for [`WeightedSum`](@ref).
    """
    struct WeightedMean{W<:AbstractVector} <: AggregateMode
        weights::W
        normalize::Bool
    end
    WeightedMean(weights::AbstractVector; normalize::Bool = true) = WeightedMean(weights, normalize)
end
