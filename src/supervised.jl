Base.Broadcast.broadcastable(l::SupervisedLoss) = Ref(l)

# --------------------------------------------------------------
# aggmode support

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

# --------------------------------------------------------------

# abstract MarginLoss <: SupervisedLoss

value(loss::MarginLoss, target::Number, output::Number) = value(loss, target * output)
deriv(loss::MarginLoss, target::Number, output::Number) = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)
function value_deriv(loss::MarginLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end

# Fallback for losses that don't want to take advantage of this
value_deriv(loss::MarginLoss, agreement::Number) = (value(loss, agreement), deriv(loss, agreement))

isunivfishercons(::MarginLoss) = false
isfishercons(loss::MarginLoss) = isunivfishercons(loss)
isnemitski(::MarginLoss) = true
ismarginbased(::MarginLoss) = true

# For every convex margin based loss L(a) the following statements
# are equivalent:
#   - L is classification calibrated
#   - L is differentiable at 0 and L'(0) < 0
# We use this here as fallback implementation.
# The other direction of the implication is not implemented however.
isclasscalibrated(loss::MarginLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

# --------------------------------------------------------------

# abstract DistanceLoss <: SupervisedLoss

value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)

# Fallback for losses that don't want to take advantage of this
value_deriv(loss::DistanceLoss, difference::Number) = (value(loss, difference), deriv(loss, difference))

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false
isclipable(::DistanceLoss) = true

# --------------------------------------------------------------
# Fallback implementations and docstrings

@doc doc"""
    value(loss, target::Number, output::Number) -> Number

Compute the (non-negative) numeric result for the loss-function
denoted by the parameter `loss` and return it. Note that `target`
and `output` can be of different numeric type, in which case
promotion is performed in the manner appropriate for the given
loss.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

```math
L : Y \times \mathbb{R} \rightarrow [0,\infty)
```

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we want to
  compute the value with.

- `target::Number`: The ground truth ``y \in Y`` of the observation.

- `output::Number`: The predicted output ``\hat{y} \in \mathbb{R}``
  for the observation.

# Examples

```jldoctest
#               loss        y    ŷ
julia> value(L1DistLoss(), 1.0, 2.0)
1.0

julia> value(L1DistLoss(), 1, 2)
1

julia> value(L1HingeLoss(), -1, 2)
3

julia> value(L1HingeLoss(), -1f0, 2f0)
3.0f0
```
"""
value(l::SupervisedLoss, target::Number, output::Number) =
    MethodError(value, (l, target, output))

"""
    value(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) -> Number

Compute the weighted or unweighted sum or mean (depending on
`avgmode`) of the individual values of the loss function for each
pair in `targets` and `outputs`. This method will not allocate a
temporary array.

In the case that the two parameters are arrays with a different
number of dimensions, broadcast will be performed. Note that the
given parameters are expected to have the same size in the
dimensions they share.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

# Examples

```jldoctest
julia> value(L1DistLoss(), [1,2,3], [2,5,-2], AggMode.Sum())
9

julia> value(L1DistLoss(), [1.,2,3], [2,5,-2], AggMode.Sum())
9.0

julia> value(L1DistLoss(), [1,2,3], [2,5,-2], AggMode.Mean())
3.0

julia> value(L1DistLoss(), Float32[1,2,3], Float32[2,5,-2], AggMode.Mean())
3.0f0
```
"""
value(l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) =
    MethodError(value, (l, target, output, avgmode))

"""
    value(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> AbstractVector

Compute the values of the loss function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation (depending on
`avgmode`). This method will not allocate a temporary array, but
it will allocate the resulting vector.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. they must
not be vectors).

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. see `?ObsDim` for more information.
"""
value(l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::LearnBase.ObsDimension) =
    MethodError(value, (l, target, output, avgmode, obsdim))

"""
    value!(buffer::AbstractArray, loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> buffer

Compute the values of the loss function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation, depending on
`avgmode`. The results are stored into the given vector `buffer`.
This method will not allocate a temporary array.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. so they
must not be vectors).

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `buffer::AbstractArray`: Array to store the computed values in.
  Old values will be overwritten and lost.

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. see `?ObsDim` for more information.

# Examples

```jldoctest
julia> targets = reshape(1:8, (2, 4)) ./ 8;

julia> outputs = reshape(1:2:16, (2, 4)) ./ 8;

julia> buffer = zeros(2);

julia> value!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.First())
2-element Array{Float64,1}:
 1.5
 2.0

julia> buffer = zeros(4);

julia> value!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.Last())
4-element Array{Float64,1}:
 0.125
 0.625
 1.125
 1.625
```
"""
value!(buffer::AbstractArray, l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::LearnBase.ObsDimension) =
    MethodError(value!, (buffer, l, target, output, avgmode, obsdim))

@doc doc"""
    deriv(loss, target::Number, output::Number) -> Number

Compute the derivative for the loss-function (denoted by the
parameter `loss`) in respect to the `output`. Note that
`target` and `output` can be of different numeric type,
in which case promotion is performed in the manner
appropriate for the given loss.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we want to
  compute the derivative with.

- `target::Number`: The ground truth ``y \in Y`` of the observation.

- `output::Number`: The predicted output ``\hat{y} \in \mathbb{R}``
  for the observation.

# Examples

```jldoctest
#               loss        y    ŷ
julia> deriv(L2DistLoss(), 1.0, 2.0)
2.0

julia> deriv(L2DistLoss(), 1, 2)
2

julia> deriv(L2HingeLoss(), -1, 2)
6

julia> deriv(L2HingeLoss(), -1f0, 2f0)
6.0f0
```
"""
deriv(l::SupervisedLoss, target::Number, output::Number) =
    MethodError(deriv, (l, target, output))

"""
    deriv(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) -> Number

Compute the weighted or unweighted sum or mean (depending on `avgmode`)
of the individual derivatives of the loss function for each pair
in `targets` and `outputs`. This method will not allocate a
temporary array.

In the case that the two parameters are arrays with a different
number of dimensions, broadcast will be performed. Note that the
given parameters are expected to have the same size in the
dimensions they share.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

# Examples

```jldoctest
julia> deriv(L2DistLoss(), [1,2,3], [2,5,-2], AggMode.Sum())
-2

julia> deriv(L2DistLoss(), [1,2,3], [2,5,-2], AggMode.Mean())
-0.6666666666666666
```
"""
deriv(l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) =
    MethodError(deriv, (l, target, output, avgmode))

"""
    deriv(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> AbstractVector

Compute the derivative of the loss function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation (depending on
`avgmode`). This method will not allocate a temporary array, but
it will allocate the resulting vector.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. they must
not be vectors).

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. see `?ObsDim` for more information.
"""
deriv(l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::LearnBase.ObsDimension) =
    MethodError(deriv, (l, target, output, avgmode, obsdim))

"""
    deriv!(buffer::AbstractArray, loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> buffer

Compute the derivative of the loss function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation, depending on
`avgmode`. The results are stored into the given vector `buffer`.
This method will not allocate a temporary array.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. so they
must not be vectors).

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `buffer::AbstractArray`: Array to store the computed values in.
  Old values will be overwritten and lost.

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. see `?ObsDim` for more information.

# Examples

```jldoctest
julia> targets = reshape(1:8, (2, 4)) ./ 8;

julia> outputs = reshape(1:2:16, (2, 4)) ./ 8;

julia> buffer = zeros(2);

julia> deriv!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.First())
2-element Array{Float64,1}:
 3.0
 4.0

julia> buffer = zeros(4);

julia> deriv!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.Last())
4-element Array{Float64,1}:
 1.0
 2.0
 2.0
 2.0
```
"""
deriv!(buffer::AbstractArray, l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::LearnBase.ObsDimension) =
    MethodError(deriv!, (buffer, l, target, output, avgmode, obsdim))

@doc doc"""
    deriv2(loss, target::Number, output::Number) -> Number

Compute the second derivative for the loss-function (denoted by
the parameter `loss`) in respect to the `output`. Note that
`target` and `output` can be of different numeric type, in which
case promotion is performed in the manner appropriate for the
given loss.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we want to
  compute the second derivative with.

- `target::Number`: The ground truth ``y \in Y`` of the observation.

- `output::Number`: The predicted output ``\hat{y} \in \mathbb{R}``
  for the observation.

# Examples

```jldoctest
#               loss             y    ŷ
julia> deriv2(LogitDistLoss(), -0.5, 0.3)
0.42781939304058886

julia> deriv2(LogitMarginLoss(), -1f0, 2f0)
0.104993574f0
```
"""
deriv2(l::SupervisedLoss, target::Number, output::Number) =
    MethodError(deriv2, (l, target, output))

"""
    deriv2(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) -> Number

Compute the weighted or unweighted sum or mean (depending on `avgmode`)
of the individual second derivatives of the loss function for
each pair in `targets` and `outputs`. This method will not
allocate a temporary array.

In the case that the two parameters are arrays with a different
number of dimensions, broadcast will be performed. Note that the
given parameters are expected to have the same size in the
dimensions they share.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

# Examples

```jldoctest
julia> deriv2(LogitDistLoss(), [1.,2,3], [2,5,-2], AggMode.Sum())
0.49687329928636825

julia> deriv2(LogitDistLoss(), [1.,2,3], [2,5,-2], AggMode.Mean())
0.1656244330954561
```
"""
deriv2(l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) =
    MethodError(deriv2, (l, target, output, avgmode))

"""
    deriv2(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> AbstractVector

Compute the second derivative of the loss function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation (depending on
`avgmode`). This method will not allocate a temporary array, but
it will allocate the resulting vector.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. they must
not be vectors).

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. see `?ObsDim` for more information.
"""
deriv2(l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::LearnBase.ObsDimension) =
    MethodError(deriv2, (l, target, output, avgmode, obsdim))

"""
    deriv2!(buffer::AbstractArray, loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> buffer

Compute the second derivative of the loss function for each pair in
`targets` and `outputs` individually, and return either the
weighted or unweighted sum or mean for each observation, depending on
`avgmode`. The results are stored into the given vector `buffer`.
This method will not allocate a temporary array.

Both arrays have to be of the same shape and size. Furthermore
they have to have at least two array dimensions (i.e. so they
must not be vectors).

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `buffer::AbstractArray`: Array to store the computed values in.
  Old values will be overwritten and lost.

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `targets::AbstractArray`: The array of ground truths ``\\mathbf{y}``.

- `outputs::AbstractArray`: The array of predicted outputs ``\\mathbf{\\hat{y}}``.

- `avgmode::AggregateMode`: Must be one of the following: [`AggMode.Sum()`](@ref),
  [`AggMode.Mean()`](@ref), [`AggMode.WeightedSum`](@ref), or
  [`AggMode.WeightedMean`](@ref).

- `obsdim::ObsDimension`: Specifies which of the array dimensions
  denotes the observations. see `?ObsDim` for more information.

# Examples

```jldoctest
julia> targets = reshape(1:8, (2, 4)) ./ 8;

julia> outputs = reshape(1:2:16, (2, 4)) ./ 8;

julia> buffer = zeros(2);

julia> deriv2!(buffer, L2DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.First())
2-element Array{Float64,1}:
 8.0
 8.0

julia> buffer = zeros(4);

julia> deriv2!(buffer, L2DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.Last())
4-element Array{Float64,1}:
 4.0
 4.0
 4.0
 4.0
```
"""
deriv2!(buffer::AbstractArray, l::Loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::LearnBase.ObsDimension) =
    MethodError(deriv2!, (buffer, l, target, output, avgmode, obsdim))

@doc doc"""
    value_deriv(loss, target::Number, output::Number) -> Tuple

Return the results of [`value`](@ref) and [`deriv`](@ref) as a
tuple, in which the first element is the value and the second
element the derivative.

In some cases this function can yield better performance, because
the losses can make use of shared variables when computing
the results. Note that `target` and `output` can be of
different numeric type, in which case promotion is performed
in the manner appropriate for the given loss.

Note: This function should always be type-stable. If it isn't,
you likely found a bug.

# Arguments

- `loss::SupervisedLoss`: The loss-function ``L`` we are working with.

- `target::Number`: The ground truth ``y \in Y`` of the observation.

- `output::Number`: The predicted output ``\hat{y} \in \mathbb{R}``


# Examples

```jldoctest
#                     loss         y    ŷ
julia> value_deriv(L2DistLoss(), -1.0, 3.0)
(16.0, 8.0)
```
"""
function value_deriv(l::SupervisedLoss, target::Number, output::Number)
    value(l, target, output), deriv(l, target, output)
end

@doc doc"""
    isconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a convex function.
A function ``f : \mathbb{R}^n \rightarrow \mathbb{R}`` is convex if
its domain is a convex set and if for all ``x, y`` in that
domain, with ``\theta`` such that for ``0 \leq \theta \leq 1``,
we have

```math
f(\theta x + (1 - \theta) y) \leq \theta f(x) + (1 - \theta) f(y)
```

# Examples

```jldoctest
julia> isconvex(LPDistLoss(0.5))
false

julia> isconvex(ZeroOneLoss())
false

julia> isconvex(L1DistLoss())
true

julia> isconvex(L2DistLoss())
true
```
"""
isconvex

@doc doc"""
    isstrictlyconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a strictly convex function.
A function ``f : \mathbb{R}^n \rightarrow \mathbb{R}`` is
strictly convex if its domain is a convex set and if for all
``x, y`` in that domain where ``x \neq y``, with
``\theta`` such that for ``0 < \theta < 1``, we have

```math
f(\theta x + (1 - \theta) y) < \theta f(x) + (1 - \theta) f(y)
```

# Examples

```jldoctest
julia> isstrictlyconvex(L1DistLoss())
false

julia> isstrictlyconvex(LogitDistLoss())
true

julia> isstrictlyconvex(L2DistLoss())
true
```
"""
isstrictlyconvex

@doc doc"""
    isstronglyconvex(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a strongly convex function.
A function ``f : \mathbb{R}^n \rightarrow \mathbb{R}`` is
``m``-strongly convex if its domain is a convex set and if
``\forall x,y \in`` **dom** ``f`` where ``x \neq y``,
and ``\theta`` such that for ``0 \le \theta \le 1`` , we have

```math
f(\theta x + (1 - \theta)y) < \theta f(x) + (1 - \theta) f(y) - 0.5 m \cdot \theta (1 - \theta) {\| x - y \|}_2^2
```

In a more familiar setting, if the loss function is
differentiable we have

```math
\left( \nabla f(x) - \nabla f(y) \right)^\top (x - y) \ge m {\| x - y\|}_2^2
```

# Examples

```jldoctest
julia> isstronglyconvex(L1DistLoss())
false

julia> isstronglyconvex(LogitDistLoss())
false

julia> isstronglyconvex(L2DistLoss())
true
```
"""
isstronglyconvex(::SupervisedLoss) = false

@doc doc"""
    isdifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function ``f : \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}`` is
differentiable at a point ``x \in`` **int dom** ``f``, if there
exists a matrix ``Df(x) \in \mathbb{R}^{m \times n}`` such that
it satisfies:

```math
\lim_{z \neq x, z \to x} \frac{{\|f(z) - f(x) - Df(x)(z-x)\|}_2}{{\|z - x\|}_2} = 0
```

A function is differentiable if its domain is open and it is
differentiable at every point ``x``.

# Examples

```jldoctest
julia> isdifferentiable(L1DistLoss())
false

julia> isdifferentiable(L1DistLoss(), 1)
true

julia> isdifferentiable(L2DistLoss())
true
```
"""
isdifferentiable(l::SupervisedLoss) = istwicedifferentiable(l)
isdifferentiable(l::SupervisedLoss, at) = isdifferentiable(l)

@doc doc"""
    istwicedifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool

Return `true` if the given `loss` is differentiable
(optionally limited to the given point `x` if specified).

A function ``f : \mathbb{R}^{n} \rightarrow \mathbb{R}`` is
said to be twice differentiable at a point ``x \in`` **int
dom** ``f``, if the function derivative for ``\nabla f``
exists at ``x``.

```math
\nabla^2 f(x) = D \nabla f(x)
```

A function is twice differentiable if its domain is open and it
is twice differentiable at every point ``x``.

# Examples

```jldoctest
julia> isdifferentiable(L1DistLoss())
false

julia> isdifferentiable(L1DistLoss(), 1)
true

julia> isdifferentiable(L2DistLoss())
true
```
"""
istwicedifferentiable(::SupervisedLoss) = false
istwicedifferentiable(l::SupervisedLoss, at) = istwicedifferentiable(l)

@doc doc"""
    islocallylipschitzcont(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is locally-Lipschitz
continous.

A supervised loss ``L : Y \times \mathbb{R} \rightarrow [0, \infty)``
is called locally Lipschitz continuous if ``\forall a \ge 0``
there exists a constant :math:`c_a \ge 0`, such that

```math
\sup_{y \in Y} \left| L(y,t) − L(y,t′) \right| \le c_a |t − t′|,  \qquad  t,t′ \in [−a,a]
```

Every convex function is locally lipschitz continuous

# Examples

```jldoctest
julia> islocallylipschitzcont(ExpLoss())
true

julia> islocallylipschitzcont(SigmoidLoss())
true
```
"""
islocallylipschitzcont(l::SupervisedLoss) = isconvex(l) || islipschitzcont(l)

@doc doc"""
    islipschitzcont(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is Lipschitz continuous.

A supervised loss function ``L : Y \times \mathbb{R} \rightarrow
[0, \infty)`` is Lipschitz continous, if there exists a finite
constant ``M < \infty`` such that

```math
|L(y, t) - L(y, t′)| \le M |t - t′|,  \qquad  \forall (y, t) \in Y \times \mathbb{R}
```

# Examples

```jldoctest
julia> islipschitzcont(SigmoidLoss())
true

julia> islipschitzcont(ExpLoss())
false
```
"""
islipschitzcont(::SupervisedLoss) = false

@doc doc"""
    isnemitski(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` denotes a Nemitski loss function.

We call a supervised loss function ``L : Y \times \mathbb{R}
\rightarrow [0,\infty)`` a Nemitski loss if there exist a
measurable function ``b : Y \rightarrow [0, \infty)`` and an
increasing function ``h : [0, \infty) \rightarrow [0, \infty)``
such that

```math
L(y,\hat{y}) \le b(y) + h(|\hat{y}|),  \qquad  (y, \hat{y}) \in Y \times \mathbb{R}.
```

If a loss if locally lipsschitz continuous then it is a Nemitski loss
"""
isnemitski(l::SupervisedLoss) = islocallylipschitzcont(l)

@doc doc"""
    isclipable(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` function is clipable. A
supervised loss ``L : Y \times \mathbb{R} \rightarrow [0,
\infty)`` can be clipped at ``M > 0`` if, for all ``(y,t)
\in Y \times \mathbb{R}``,

```math
L(y, \hat{t}) \le L(y, t)
```

where ``\hat{t}`` denotes the clipped value of ``t`` at
``\pm M``. That is

```math
\hat{t} = \begin{cases} -M & \quad \text{if } t < -M \\ t & \quad \text{if } t \in [-M, M] \\ M & \quad \text{if } t > M \end{cases}
```

# Examples

```jldoctest
julia> isclipable(ExpLoss())
false

julia> isclipable(L2DistLoss())
true
```
"""
isclipable(::SupervisedLoss) = false

@doc doc"""
    ismarginbased(loss::SupervisedLoss) -> Bool

Return `true` if the given `loss` is a margin-based loss.

A supervised loss function ``L : Y \times \mathbb{R} \rightarrow
[0, \infty)`` is said to be **margin-based**, if there exists a
representing function ``\psi : \mathbb{R} \rightarrow [0, \infty)``
satisfying

```math
L(y, \hat{y}) = \psi (y \cdot \hat{y}),  \qquad  (y, \hat{y}) \in Y \times \mathbb{R}
```

# Examples

```jldoctest
julia> ismarginbased(HuberLoss(2))
false

julia> ismarginbased(L2MarginLoss())
true
```
"""
ismarginbased(::SupervisedLoss) = false

"""
    isclasscalibrated(loss::SupervisedLoss) -> Bool
"""
isclasscalibrated(::SupervisedLoss) = false

@doc doc"""
    isdistancebased(loss::SupervisedLoss) -> Bool

Return `true` ifthe given `loss` is a distance-based loss.

A supervised loss function ``L : Y \times \mathbb{R} \rightarrow
[0, \infty)`` is said to be **distance-based**, if there exists a
representing function ``\psi : \mathbb{R} \rightarrow [0, \infty)``
satisfying ``\psi (0) = 0`` and

```math
L(y, \hat{y}) = \psi (\hat{y} - y),  \qquad  (y, \hat{y}) \in Y \times \mathbb{R}
```

# Examples

```jldoctest
julia> isdistancebased(HuberLoss(2))
true

julia> isdistancebased(L2MarginLoss())
false
```
"""
isdistancebased(::SupervisedLoss) = false

@doc doc"""
    issymmetric(loss::SupervisedLoss) -> Bool

Return `true` if the given loss is a symmetric loss.

A function ``f : \mathbb{R} \rightarrow [0,\infty)`` is said
to be symmetric about origin if we have

```math
f(x) = f(-x), \qquad  \forall x \in \mathbb{R}
```

A distance-based loss is said to be symmetric if its representing
function is symmetric.

# Examples

```jldoctest
julia> issymmetric(QuantileLoss(0.2))
false

julia> issymmetric(LPDistLoss(2))
true
```
"""
issymmetric(::SupervisedLoss) = false

isminimizable(l::SupervisedLoss) = isconvex(l)

# -----------------
# IMPLEMENTATIONS
# -----------------
include("supervised/sparse.jl")
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/scaled.jl")
include("supervised/weightedbinary.jl")
include("supervised/other.jl")
include("supervised/ordinal.jl")
