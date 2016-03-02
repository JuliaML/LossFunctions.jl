
abstract Loss
abstract ModelLoss <: Loss

@inline call(c::ModelLoss, X, target, α) = value(c, X, target, α)
@inline transpose(c::ModelLoss) = deriv_fun(c)

@inline function value_fun(c::ModelLoss)
    @inline _value(args...) = value(c, args...)
    _value
end

@inline function deriv_fun(c::ModelLoss)
    @inline _deriv(args...) = deriv(c, args...)
    _deriv
end

@inline function deriv2_fun(c::ModelLoss)
    @inline _deriv2(args...) = deriv2(c, args...)
    _deriv2
end

@inline function value_deriv_fun(c::ModelLoss)
    @inline _value_deriv(args...) = value_deriv(c, args...)
    _value_deriv
end

isminimizable(c::ModelLoss) = isconvex(c)
isdifferentiable(c::ModelLoss) = istwicedifferentiable(c)
istwicedifferentiable(::ModelLoss) = false
isdifferentiable(c::ModelLoss, at) = isdifferentiable(c)
istwicedifferentiable(c::ModelLoss, at) = istwicedifferentiable(c)
isconvex(::ModelLoss) = false
isstronglyconvex(::ModelLoss) = false

# ==========================================================================

isnemitski(loss::ModelLoss) = islocallylipschitzcont(loss)
islipschitzcont(::ModelLoss) = false
islocallylipschitzcont(::ModelLoss) = false
isclipable(::ModelLoss) = false
islipschitzcont_deriv(::ModelLoss) = false

# ==========================================================================

@inline call(loss::ModelLoss, target, output) = value(loss, target, output)
@inline value(loss::ModelLoss, X::AbstractArray, target::Number, output::Number) = value(loss, target, output)
@inline deriv(loss::ModelLoss, X::AbstractArray, target::Number, output::Number) = deriv(loss, target, output)
@inline deriv2(loss::ModelLoss, X::AbstractArray, target::Number, output::Number) = deriv2(loss, target, output)
@inline value_deriv(loss::ModelLoss, X::AbstractArray, target::Number, output::Number) = value_deriv(loss, target, output)

# --------------------------------------------------------------------------

@inline function value(loss::ModelLoss, target::AbstractVecOrMat, output::AbstractVecOrMat)
    buffer = similar(output)
    value!(buffer, loss, target, output)
end

@inline function deriv(loss::ModelLoss, target::AbstractVector, output::AbstractVecOrMat)
    buffer = similar(output)
    deriv!(buffer, loss, target, output)
end

@inline function grad(loss::ModelLoss, target::AbstractMatrix, output::AbstractVecOrMat)
    buffer = similar(output)
    grad!(buffer, loss, target, output)
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractVector, loss::ModelLoss, target::AbstractVector, output::AbstractVector)
    n = length(output)
    @_dimcheck length(target) == n && size(buffer) == size(output)
    @simd for i = 1:n
        @inbounds buffer[i] = value(loss, target[i], output[i])
    end
    buffer
end

@inline function deriv!(buffer::AbstractVector, loss::ModelLoss, target::AbstractVector, output::AbstractVector)
    n = length(output)
    @_dimcheck length(target) == n && size(buffer) == size(output)
    @simd for i = 1:n
        @inbounds buffer[i] = deriv(loss, target[i], output[i])
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractMatrix, loss::ModelLoss, target::AbstractVector, output::AbstractMatrix)
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

@inline function deriv!(buffer::AbstractMatrix, loss::ModelLoss, target::AbstractVector, output::AbstractMatrix)
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

@inline function value!(buffer::AbstractMatrix, loss::ModelLoss, target::AbstractMatrix, output::AbstractMatrix)
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

@inline function grad!(buffer::AbstractMatrix, loss::ModelLoss, target::AbstractMatrix, output::AbstractMatrix)
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

@inline function sumvalue{T<:Number}(loss::ModelLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += value(loss, target[i], output[i])
    end
    val
end

@inline function sumderiv{T<:Number}(loss::ModelLoss, target::AbstractVector, output::AbstractArray{T})
    n = length(output)
    @_dimcheck length(target) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += deriv(loss, target[i], output[i])
    end
    val
end

# --------------------------------------------------------------------------

@inline function meanvalue{T<:Number}(loss::ModelLoss, target::AbstractVector, output::AbstractArray{T})
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

@inline function meanderiv{T<:Number}(loss::ModelLoss, target::AbstractVector, output::AbstractArray{T})
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

ismarginbased(::ModelLoss) = false
isclasscalibrated(::ModelLoss) = false
isdistancebased(::ModelLoss) = false
issymmetric(::ModelLoss) = false

# ==========================================================================

abstract MarginLoss <: ModelLoss

@inline call(loss::MarginLoss, agreement) = value(loss, agreement)
@inline value(loss::MarginLoss, target::Number, output::Number) = value(loss, target * output)
@inline deriv(loss::MarginLoss, target::Number, output::Number) = target * deriv(loss, target * output)
@inline deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)
@inline function value_deriv(loss::MarginLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end

isunivfishercons(::MarginLoss) = false
isfishercons(loss::MarginLoss) = isunivfishercons(loss)
isnemitski(::MarginLoss) = true
islocallylipschitzcont(loss::MarginLoss) = isconvex(loss)
ismarginbased(::MarginLoss) = true
isclasscalibrated(loss::MarginLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

# ==========================================================================

abstract DistanceLoss <: ModelLoss

@inline call(loss::DistanceLoss, residual) = value(loss, residual)
@inline value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
@inline deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
@inline deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
@inline value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false

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
