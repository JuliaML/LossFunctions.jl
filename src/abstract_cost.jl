abstract PredictionLoss

@inline call(c::PredictionLoss, X, y, α) = value(c, X, y, α)
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

isnemitski(l::PredictionLoss) = islocallylipschitzcont(l)
islipschitzcont(::PredictionLoss) = false
islocallylipschitzcont(::PredictionLoss) = false
isclipable(::PredictionLoss) = false
islipschitzcont_deriv(::PredictionLoss) = false

# ==========================================================================

@inline call(l::PredictionLoss, y, t) = value(l, y, t)
@inline value(l::PredictionLoss, X::AbstractArray, y::Number, t::Number) = value(l, y, t)
@inline deriv(l::PredictionLoss, X::AbstractArray, y::Number, t::Number) = deriv(l, y, t)
@inline deriv2(l::PredictionLoss, X::AbstractArray, y::Number, t::Number) = deriv2(l, y, t)
@inline value_deriv(l::PredictionLoss, X::AbstractArray, y::Number, t::Number) = value_deriv(l, y, t)

# --------------------------------------------------------------------------

@inline function value(l::PredictionLoss, y::AbstractVecOrMat, t::AbstractVecOrMat)
    buffer = similar(t)
    value!(buffer, l, y, t)
end

@inline function deriv(l::PredictionLoss, y::AbstractVector, t::AbstractVecOrMat)
    buffer = similar(t)
    deriv!(buffer, l, y, t)
end

@inline function grad(l::PredictionLoss, y::AbstractMatrix, t::AbstractVecOrMat)
    buffer = similar(t)
    grad!(buffer, l, y, t)
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractVector, l::PredictionLoss, y::AbstractVector, t::AbstractVector)
    n = length(t)
    @_dimcheck length(y) == n && size(buffer) == size(t)
    @simd for i = 1:n
        @inbounds buffer[i] = value(l, y[i], t[i])
    end
    buffer
end

@inline function deriv!(buffer::AbstractVector, l::PredictionLoss, y::AbstractVector, t::AbstractVector)
    n = length(t)
    @_dimcheck length(y) == n && size(buffer) == size(t)
    @simd for i = 1:n
        @inbounds buffer[i] = deriv(l, y[i], t[i])
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractMatrix, l::PredictionLoss, y::AbstractVector, t::AbstractMatrix)
    n = size(t, 2)
    k = size(t, 1)
    @_dimcheck length(y) == n && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = value(l, y[i], t[j, i])
        end
    end
    buffer
end

@inline function deriv!(buffer::AbstractMatrix, l::PredictionLoss, y::AbstractVector, t::AbstractMatrix)
    n = size(t, 2)
    k = size(t, 1)
    @_dimcheck length(y) == n && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = deriv(l, y[i], t[j, i])
        end
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractMatrix, l::PredictionLoss, y::AbstractMatrix, t::AbstractMatrix)
    n = size(t, 2)
    k = size(t, 1)
    @_dimcheck size(y) == size(t) && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = value(l, y[j, i], t[j, i])
        end
    end
    buffer
end

@inline function grad!(buffer::AbstractMatrix, l::PredictionLoss, y::AbstractMatrix, t::AbstractMatrix)
    n = size(t, 2)
    k = size(t, 1)
    @_dimcheck size(y) == size(t) && size(buffer) == (k, n)
    for i = 1:n
        @simd for j = 1:k
            @inbounds buffer[j, i] = deriv(l, y[j, i], t[j, i])
        end
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function sumvalue{T<:Number}(l::PredictionLoss, y::AbstractVector, t::AbstractArray{T})
    n = length(t)
    @_dimcheck length(y) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += value(l, y[i], t[i])
    end
    val
end

@inline function sumderiv{T<:Number}(l::PredictionLoss, y::AbstractVector, t::AbstractArray{T})
    n = length(t)
    @_dimcheck length(y) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += deriv(l, y[i], t[i])
    end
    val
end

# --------------------------------------------------------------------------

@inline function meanvalue{T<:Number}(l::PredictionLoss, y::AbstractVector, t::AbstractArray{T})
    n = length(t)
    @_dimcheck length(y) == n
    val = zero(T)
    tmp = zero(T)
    @simd for i = 1:n
        @inbounds tmp = value(l, y[i], t[i])::T
        tmp /= n
        val += tmp
    end
    val
end

@inline function meanderiv{T<:Number}(l::PredictionLoss, y::AbstractVector, t::AbstractArray{T})
    n = length(t)
    @_dimcheck length(y) == n
    val = zero(T)
    tmp = zero(T)
    @simd for i = 1:n
        @inbounds tmp = deriv(l, y[i], t[i])::T
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

@inline call(l::MarginBasedLoss, yt) = value(l, yt)
@inline value(l::MarginBasedLoss, y::Number, t::Number) = value(l, y * t)
@inline deriv(l::MarginBasedLoss, y::Number, t::Number) = y * deriv(l, y * t)
@inline deriv2(l::MarginBasedLoss, y::Number, t::Number) = deriv2(l, y * t)
@inline function value_deriv(l::MarginBasedLoss, y::Number, t::Number)
    v, d = value_deriv(l, y * t)
    (v, y*d)
end

isunivfishercons(::MarginBasedLoss) = false
isfishercons(l::MarginBasedLoss) = isunivfishercons(l)
isnemitski(::MarginBasedLoss) = true
islocallylipschitzcont(l::MarginBasedLoss) = isconvex(l)
ismarginbased(::MarginBasedLoss) = true
isclasscalibrated(l::MarginBasedLoss) = isconvex(l) && isdifferentiable(l, 0) && deriv(l, 0) < 0

# ==========================================================================

abstract DistanceBasedLoss <: PredictionLoss

@inline call(l::DistanceBasedLoss, r) = value(l, r)
@inline value(l::DistanceBasedLoss, y::Number, t::Number) = value(l, t - y)
@inline deriv(l::DistanceBasedLoss, y::Number, t::Number) = deriv(l, t - y)
@inline deriv2(l::DistanceBasedLoss, y::Number, t::Number) = deriv2(l, t - y)
@inline value_deriv(l::DistanceBasedLoss, y::Number, t::Number) = value_deriv(l, t - y)

isdistancebased(::DistanceBasedLoss) = true
issymmetric(::DistanceBasedLoss) = false
