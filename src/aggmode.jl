abstract type AggregateMode end

module AggMode

    using StatsBase
    import ..LossFunctions.AggregateMode

    export
        None,
        Sum,
        Mean,
        WeightedMean,
        WeightedSum

    """
        AggMode.None()

    Opt-out of aggregation. This is usually the default value.
    Using `None` will cause the element-wise results to be returned.
    """
    struct None <: AggregateMode end

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

    # Examples

    ```julia-repl
    julia> AggMode.WeightedSum([1,1,2]); # 3 observations

    julia> AggMode.WeightedSum([1,1,2], normalize = true);
    ```
    """
    struct WeightedSum{T<:AbstractWeights} <: AggregateMode
        weights::T
        normalize::Bool
    end
    WeightedSum(A::AbstractVector, normalize::Bool) = WeightedSum(weights(A), normalize)
    WeightedSum(A::AbstractVector; normalize::Bool = false) = WeightedSum(weights(A), normalize)

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

    # Examples

    ```julia-repl
    julia> AggMode.WeightedMean([1,1,2]); # 3 observations

    julia> AggMode.WeightedMean([1,1,2], normalize = false);
    ```
    """
    struct WeightedMean{T<:AbstractWeights} <: AggregateMode
        weights::T
        normalize::Bool
    end
    WeightedMean(A::AbstractVector, normalize::Bool) = WeightedMean(weights(A), normalize)
    WeightedMean(A::AbstractVector; normalize::Bool = true) = WeightedMean(weights(A), normalize)
end

for (FUN, DESC, EXAMPLE) in (
        (:value,
         "result of the loss function",
         """
         3-element Array{Float64,1}:
           1.0
           9.0
          25.0
         """
        ),
        (:deriv,
         "derivative of the loss function in respect to the output",
         """
         3-element Array{Float64,1}:
            2.0
            6.0
          -10.0
         """
        ),
        (:deriv2,
         "second derivative of the loss function in respect to the output",
         """
         3-element Array{Float64,1}:
          2.0
          2.0
          2.0
         """
        ))
    @eval begin

        # ------------------------------------------------------
        # FALLBACK

        # By default compute the element-wise result
        @doc """
            $($(string(FUN)))(loss, targets::AbstractArray, outputs::AbstractArray)

        Compute the $($(DESC)) for each index-pair in `targets`
        and `outputs` individually and return the result as an
        array of the appropriate size.

        In the case that the two parameters are arrays with a
        different number of dimensions, broadcast will be
        performed. Note that the given parameters are expected to
        have the same size in the dimensions they share.

        Note: This function should always be type-stable. If it
        isn't, you likely found a bug.

        # Arguments

        - `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

        - `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

        - `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

        # Examples

        ```jldoctest
        julia> $($(string(FUN)))(L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])
        $($(EXAMPLE))
        ```
        """
        @inline function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray)
            ($FUN)(loss, target, output, AggMode.None())
        end

        # (mutating) By default compute the element-wise result
        @doc """
            $($(string(FUN)))!(buffer::AbstractArray, loss, targets::AbstractArray, outputs::AbstractArray) -> buffer

        Compute the $($(DESC)) for each index-pair in `targets`
        and `outputs` individually, and store them in the
        preallocated `buffer`. Note that `buffer` has to be of
        the appropriate size.

        In the case that the two parameters, `targets` and
        `outputs`, are arrays with a different number of
        dimensions, broadcast will be performed. Note that the
        given parameters are expected to have the same size in
        the dimensions they share.

        Note: This function should always be type-stable. If it
        isn't, you likely found a bug.

        # Arguments

        - `buffer::AbstractArray`: Array to store the computed
          values in. Old values will be overwritten and lost.

        - `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

        - `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

        - `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

        # Examples

        ```jldoctest
        julia> buffer = zeros(3); # preallocate a buffer

        julia> $($(string(FUN)))!(buffer, L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])
        $($(EXAMPLE))
        ```
        """
        @inline function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray)
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, AggMode.None())
        end

        # Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        @inline function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray{T,N},
                agg::AggregateMode,
                ::ObsDim.Last = ObsDim.Last()) where {T,N}
            ($FUN)(loss, target, output, agg, ObsDim.Constant{N}())
        end

        # (mutating) Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        @inline function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray{T,N},
                agg::AggregateMode,
                ::ObsDim.Last = ObsDim.Last()) where {T,N}
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, agg, ObsDim.Constant{N}())
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin

            # --------------------------------------------------
            # FALLBACK

            # By default compute the element-wise result
            @inline function ($FUN)(loss::$KIND, numbers::AbstractArray)
                ($FUN)(loss, numbers, AggMode.None())
            end

            # (mutating) By default compute the element-wise result
            @inline function ($(Symbol(FUN,:!)))(
                    buffer::AbstractArray,
                    loss::$KIND,
                    numbers::AbstractArray)
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, AggMode.None())
            end

            # Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
            @inline function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    agg::AggregateMode,
                    ::ObsDim.Last = ObsDim.Last()) where {T,N}
                ($FUN)(loss, numbers, agg, ObsDim.Constant{N}())
            end

            # (mutating) Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
            @inline function ($(Symbol(FUN,:!)))(
                    buffer::AbstractArray,
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    agg::AggregateMode,
                    ::ObsDim.Last = ObsDim.Last()) where {T,N}
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, agg, ObsDim.Constant{N}())
            end
        end
    end
end

# -----------------
# IMPLEMENTATIONS
# -----------------
include("aggmode/none.jl")
include("aggmode/sum.jl")
include("aggmode/wsum.jl")
include("aggmode/mean.jl")
include("aggmode/wmean.jl")
