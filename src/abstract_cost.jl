abstract Cost

@inline call(c::Cost, X, y, α) = value(c, X, y, α)
@inline transpose(c::Cost) = deriv_fun(c)

@inline function value_fun(c::Cost)
    @inline _value(args...) = value(c, args...)
    _value
end

@inline function deriv_fun(c::Cost)
    @inline _deriv(args...) = deriv(c, args...)
    _deriv
end

@inline function deriv2_fun(c::Cost)
    @inline _deriv2(args...) = deriv2(c, args...)
    _deriv2
end

@inline function value_deriv_fun(c::Cost)
    @inline _value_deriv(args...) = value_deriv(c, args...)
    _value_deriv
end

isminimizable(c::Cost) = isconvex(c)
isdifferentiable(c::Cost) = istwicedifferentiable(c)
istwicedifferentiable(::Cost) = false
isdifferentiable(c::Cost, at) = isdifferentiable(c)
istwicedifferentiable(c::Cost, at) = istwicedifferentiable(c)
isconvex(::Cost) = false

# ==========================================================================

abstract Loss <: Cost

isnemitski(l::Loss) = islocallylipschitzcont(l)
islipschitzcont(::Loss) = false
islocallylipschitzcont(::Loss) = false
isclipable(::Loss) = false

# ==========================================================================

abstract SupervisedLoss <: Loss

@inline call(l::SupervisedLoss, y, t) = value(l, y, t)
@inline value(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = value(l, y, t)
@inline deriv(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv(l, y, t)
@inline deriv2(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv2(l, y, t)
@inline value_deriv(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = value_deriv(l, y, t)

# --------------------------------------------------------------------------

@inline function value(l::SupervisedLoss, y::AbstractVecOrMat, t::AbstractVecOrMat)
    buffer = similar(t)
    value!(buffer, l, y, t)
end

@inline function deriv(l::SupervisedLoss, y::AbstractVector, t::AbstractVecOrMat)
    buffer = similar(t)
    deriv!(buffer, l, y, t)
end

@inline function grad(l::SupervisedLoss, y::AbstractMatrix, t::AbstractVecOrMat)
    buffer = similar(t)
    grad!(buffer, l, y, t)
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractVector, l::SupervisedLoss, y::AbstractVector, t::AbstractVector)
    n = length(t)
    @_dimcheck length(y) == n && size(buffer) == size(t)
    @simd for i = 1:n
        @inbounds buffer[i] = value(l, y[i], t[i])
    end
    buffer
end

@inline function deriv!(buffer::AbstractVector, l::SupervisedLoss, y::AbstractVector, t::AbstractVector)
    n = length(t)
    @_dimcheck length(y) == n && size(buffer) == size(t)
    @simd for i = 1:n
        @inbounds buffer[i] = deriv(l, y[i], t[i])
    end
    buffer
end

# --------------------------------------------------------------------------

@inline function value!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractVector, t::AbstractMatrix)
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

@inline function deriv!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractVector, t::AbstractMatrix)
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

@inline function value!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractMatrix, t::AbstractMatrix)
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

@inline function grad!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractMatrix, t::AbstractMatrix)
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

@inline function sumvalue{T<:Number}(l::SupervisedLoss, y::AbstractVector, t::AbstractArray{T})
    n = length(t)
    @_dimcheck length(y) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += value(l, y[i], t[i])
    end
    val
end

@inline function sumderiv{T<:Number}(l::SupervisedLoss, y::AbstractVector, t::AbstractArray{T})
    n = length(t)
    @_dimcheck length(y) == n
    val = zero(T)
    @simd for i = 1:n
        @inbounds val += deriv(l, y[i], t[i])
    end
    val
end

# --------------------------------------------------------------------------

@inline function meanvalue{T<:Number}(l::SupervisedLoss, y::AbstractVector, t::AbstractArray{T})
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

@inline function meanderiv{T<:Number}(l::SupervisedLoss, y::AbstractVector, t::AbstractArray{T})
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

ismarginbased(::SupervisedLoss) = false
isclasscalibrated(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false

# ==========================================================================

abstract MarginBasedLoss <: SupervisedLoss

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

abstract DistanceBasedLoss <: SupervisedLoss

@inline call(l::DistanceBasedLoss, r) = value(l, r)
@inline value(l::DistanceBasedLoss, y::Number, t::Number) = value(l, t - y)
@inline deriv(l::DistanceBasedLoss, y::Number, t::Number) = deriv(l, t - y)
@inline deriv2(l::DistanceBasedLoss, y::Number, t::Number) = deriv2(l, t - y)
@inline value_deriv(l::DistanceBasedLoss, y::Number, t::Number) = value_deriv(l, t - y)

isdistancebased(::DistanceBasedLoss) = true
issymmetric(::DistanceBasedLoss) = false

# ==========================================================================

abstract UnsupervisedLoss <: Loss

@inline value(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = value(l, X, t)
@inline deriv(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv(l, X, t)
@inline deriv2(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv2(l, X, t)
@inline value_deriv(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = value_deriv(l, X, t)
