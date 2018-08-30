abstract type AverageMode end

module AvgMode

    using StatsBase
    import ..LossFunctions.AverageMode

    export
        None,
        Sum,
        Mean,
        Macro,
        Micro,
        WeightedMean,
        WeightedSum

    """
        AvgMode.None()

    Opt-out of aggregation. This is usually the default value.
    Using `None` will cause the element-wise results to be returned.
    """
    struct None <: AverageMode end

    """
        AvgMode.Sum()

    Causes the method to return the unweighted sum of the
    elements instead of the individual elements. Can be used in
    combination with `ObsDim`, in which case a vector will be
    returned containing the sum for each observation (useful
    mainly for multivariable regression).
    """
    struct Sum <: AverageMode end

    """
        AvgMode.Mean()

    Causes the method to return the unweighted mean of the
    elements instead of the individual elements. Can be used in
    combination with `ObsDim`, in which case a vector will be
    returned containing the mean for each observation (useful
    mainly for multivariable regression).
    """
    struct Mean <: AverageMode end

    struct Macro <: AverageMode end
    const Micro = Mean

    """
        AvgMode.WeightedSum(weights; [normalize = false])

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

    # Examples

    ```julia-repl
    julia> AvgMode.WeightedSum([1,1,2]); # 3 observations

    julia> AvgMode.WeightedSum([1,1,2], normalize = true);
    ```
    """
    struct WeightedSum{T<:AbstractWeights} <: AverageMode
        weights::T
        normalize::Bool
    end
    WeightedSum(A::AbstractVector, normalize::Bool) = WeightedSum(weights(A), normalize)
    WeightedSum(A::AbstractVector; normalize::Bool = false) = WeightedSum(weights(A), normalize)

    """
        AvgMode.WeightedMean(weights; [normalize = true])

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

    # Examples

    ```julia-repl
    julia> AvgMode.WeightedMean([1,1,2]); # 3 observations

    julia> AvgMode.WeightedMean([1,1,2], normalize = false);
    ```
    """
    struct WeightedMean{T<:AbstractWeights} <: AverageMode
        weights::T
        normalize::Bool
    end
    WeightedMean(A::AbstractVector, normalize::Bool) = WeightedMean(weights(A), normalize)
    WeightedMean(A::AbstractVector; normalize::Bool = true) = WeightedMean(weights(A), normalize)
end
