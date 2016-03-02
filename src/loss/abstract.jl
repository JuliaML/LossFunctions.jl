
abstract Loss
abstract PredictionLoss <: Loss

@inline call(c::PredictionLoss, X, target, α) = value(c, X, target, α)
@inline transpose(c::PredictionLoss) = deriv_fun(c)

@inline function value_fun(c::PredictionLoss)
    @inline _value(args...) = value(c, args...)
    _value
end

@inline function deriv_fun(c::PredictionLoss)
    @inline _deriv(args...) = deriv(c, args...)
    _deriv
end

@inline function deriv2_fun(c::PredictionLoss)
    @inline _deriv2(args...) = deriv2(c, args...)
    _deriv2
end

@inline function value_deriv_fun(c::PredictionLoss)
    @inline _value_deriv(args...) = value_deriv(c, args...)
    _value_deriv
end

isminimizable(c::PredictionLoss) = isconvex(c)
isdifferentiable(c::PredictionLoss) = istwicedifferentiable(c)
istwicedifferentiable(::PredictionLoss) = false
isdifferentiable(c::PredictionLoss, at) = isdifferentiable(c)
istwicedifferentiable(c::PredictionLoss, at) = istwicedifferentiable(c)
isconvex(::PredictionLoss) = false
isstronglyconvex(::PredictionLoss) = false

# ==========================================================================

isnemitski(loss::PredictionLoss) = islocallylipschitzcont(loss)
islipschitzcont(::PredictionLoss) = false
islocallylipschitzcont(::PredictionLoss) = false
isclipable(::PredictionLoss) = false
islipschitzcont_deriv(::PredictionLoss) = false

# ==========================================================================

@inline call(loss::PredictionLoss, target, output) = value(loss, target, output)
@inline value(loss::PredictionLoss, X::AbstractArray, target::Number, output::Number) = value(loss, target, output)
@inline deriv(loss::PredictionLoss, X::AbstractArray, target::Number, output::Number) = deriv(loss, target, output)
@inline deriv2(loss::PredictionLoss, X::AbstractArray, target::Number, output::Number) = deriv2(loss, target, output)
@inline value_deriv(loss::PredictionLoss, X::AbstractArray, target::Number, output::Number) = value_deriv(loss, target, output)

# --------------------------------------------------------------------------

@inline function value(loss::PredictionLoss, target::AbstractVecOrMat, output::AbstractVecOrMat)
    buffer = similar(output)
    value!(buffer, loss, target, output)
end

@inline function deriv(loss::PredictionLoss, target::AbstractVector, output::AbstractVecOrMat)
    buffer = similar(output)
    deriv!(buffer, loss, target, output)
end

@inline function grad(loss::PredictionLoss, target::AbstractMatrix, output::AbstractVecOrMat)
    buffer = similar(output)
    grad!(buffer, loss, target, output)
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractVector, loss::PredictionLoss, target::AbstractVector, output::AbstractVector)
    n = length(output)
    @_dimcheck length(target) == n && size(buffer) == size(output)
    @simd for i = 1:n
        @inbounds buffer[i] = value(loss, target[i], output[i])
    end
    buffer
end

@inline function deriv!(buffer::AbstractVector, loss::PredictionLoss, target::AbstractVector, output::AbstractVector)
    n = length(output)
    @_dimcheck length(target) == n && size(buffer) == size(output)
    @simd for i = 1:n
        @inbounds buffer[i] = deriv(loss, target[i], output[i])
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractMatrix, loss::PredictionLoss, target::AbstractVector, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck length(target) == n && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = value(loss, target[i], output[j, i])
        end
    end
    buffer
end

@inline function deriv!(buffer::AbstractMatrix, loss::PredictionLoss, target::AbstractVector, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck length(target) == n && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = deriv(loss, target[i], output[j, i])
        end
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractMatrix, loss::PredictionLoss, target::AbstractMatrix, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck size(target) == size(output) && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = value(loss, target[j, i], output[j, i])
        end
    end
    buffer
end

@inline function grad!(buffer::AbstractMatrix, loss::PredictionLoss, target::AbstractMatrix, output::AbstractMatrix)
    n = size(output, 2)
    k = size(output, 1)
    @_dimcheck size(target) == size(output) && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = deriv(loss, target[j, i], output[j, i])
        end
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function sumvalue{T<:Number}(loss::PredictionLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += value(loss, target[i], output[i])
    end
    val
end

@inline function sumderiv{T<:Number}(loss::PredictionLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += deriv(loss, target[i], output[i])
    end
    val
end

# --------------------------------------------------------------------------

@inline function meanvalue{T<:Number}(loss::PredictionLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    tmp = zero(T)
    @simd for i = 1:n
        @inbounds tmp = value(loss, target[i], output[i])::T
        tmp /= n
        val += tmp
    end
    val
end

@inline function meanderiv{T<:Number}(loss::PredictionLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    tmp = zero(T)
    @simd for i = 1:n
        @inbounds tmp = deriv(loss, target[i], output[i])::T
        tmp /= n
        val += tmp
    end
    val
end

ismarginbased(::PredictionLoss) = false
isclasscalibrated(::PredictionLoss) = false
isdistancebased(::PredictionLoss) = false
issymmetric(::PredictionLoss) = false

# ==========================================================================

abstract MarginBasedLoss <: PredictionLoss

@inline call(loss::MarginBasedLoss, agreement) = value(loss, agreement)
@inline value(loss::MarginBasedLoss, target::Number, output::Number) = value(loss, target * output)
@inline deriv(loss::MarginBasedLoss, target::Number, output::Number) = target * deriv(loss, target * output)
@inline deriv2(loss::MarginBasedLoss, target::Number, output::Number) = deriv2(loss, target * output)
@inline function value_deriv(loss::MarginBasedLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end

isunivfishercons(::MarginBasedLoss) = false
isfishercons(loss::MarginBasedLoss) = isunivfishercons(loss)
isnemitski(::MarginBasedLoss) = true
islocallylipschitzcont(loss::MarginBasedLoss) = isconvex(loss)
ismarginbased(::MarginBasedLoss) = true
isclasscalibrated(loss::MarginBasedLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

# ==========================================================================

abstract DistanceBasedLoss <: PredictionLoss

@inline call(loss::DistanceBasedLoss, residual) = value(loss, residual)
@inline value(loss::DistanceBasedLoss, target::Number, output::Number) = value(loss, output - target)
@inline deriv(loss::DistanceBasedLoss, target::Number, output::Number) = deriv(loss, output - target)
@inline deriv2(loss::DistanceBasedLoss, target::Number, output::Number) = deriv2(loss, output - target)
@inline value_deriv(loss::DistanceBasedLoss, target::Number, output::Number) = value_deriv(loss, output - target)

isdistancebased(::DistanceBasedLoss) = true
issymmetric(::DistanceBasedLoss) = false

# ==========================================================================


# Code based on code from OnlineStats by Josh day (see LICENSE.md)

abstract ParameterLoss <: Loss

Base.copy(loss::ParameterLoss) = deepcopy(loss)

@inline grad(loss::ParameterLoss, params::AbstractArray, len::Int=length(params)) = grad!(zeros(params), loss, params, len)

@inline function grad!{T<:Number}(buffer::AbstractArray{T}, loss::ParameterLoss, params::AbstractArray, len::Int=length(params))
    @_dimcheck length(buffer) == length(params)
    @_dimcheck 0 < len <= length(params)
    @inbounds buffer[end] = zero(T)
    @simd for j in 1:len
        @inbounds buffer[j] = deriv(loss, params[j])
    end
    buffer
end

@inline function addgrad!{T<:Number}(buffer::AbstractArray{T}, loss::ParameterLoss, params::AbstractArray, len::Int=length(params))
    @_dimcheck length(buffer) == length(params)
    @_dimcheck 0 < len <= length(params)
    @simd for j in 1:len
        @inbounds buffer[j] += deriv(loss, params[j])
    end
    buffer
end
