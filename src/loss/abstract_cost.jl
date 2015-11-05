abstract Cost

@inline call(c::Cost, X, y, α) = value(c, X, y, α)
@inline transpose(c::Cost) = deriv_fun(c)
# @inline value(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented
# @inline deriv(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented
# @inline deriv2(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented
# @inline value_deriv(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented

@inline function value_fun(c::Cost)
  _value(args...) = value(c, args...)
  _value
end

@inline function deriv_fun(c::Cost)
  _deriv(args...) = deriv(c, args...)
  _deriv
end

@inline function deriv2_fun(c::Cost)
  _deriv2(args...) = deriv2(c, args...)
  _deriv2
end

@inline function value_deriv_fun(c::Cost)
  _value_deriv(args...) = value_deriv(c, args...)
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

# @inline value(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented
# @inline deriv(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented
# @inline deriv2(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented
# @inline value_deriv(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented

@inline function value(l::SupervisedLoss, y::AbstractVecOrMat, t::AbstractVecOrMat)
  buffer = similar(t)
  value!(buffer, l, y, t)
end

@inline function grad(l::SupervisedLoss, y::AbstractVecOrMat, t::AbstractVecOrMat)
  buffer = similar(t)
  grad!(buffer, l, y, t)
end

@inline function value!(buffer::AbstractVector, l::SupervisedLoss, y::AbstractVector, t::AbstractVector)
  n = length(t)
  @_dimcheck length(y) == n && size(buffer) == size(t)
  for i = 1:n
    @inbounds buffer[i] = value(l, y[i], t[i])
  end
  buffer
end

@inline function grad!(buffer::AbstractVector, l::SupervisedLoss, y::AbstractVector, t::AbstractVector)
  n = length(t)
  @_dimcheck length(y) == n && size(buffer) == size(t)
  for i = 1:n
    @inbounds buffer[i] = deriv(l, y[i], t[i])
  end
  buffer
end

@inline function value!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractVector, t::AbstractMatrix)
  n = size(t, 2)
  k = size(t, 1)
  @_dimcheck length(y) == n && size(buffer) == (k, n)
  for i = 1:n, j = 1:k
    @inbounds buffer[j, i] = value(l, y[i], t[j, i])
  end
  buffer
end

@inline function grad!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractVector, t::AbstractMatrix)
  n = size(t, 2)
  k = size(t, 1)
  @_dimcheck length(y) == n && size(buffer) == (k, n)
  for i = 1:n, j = 1:k
    @inbounds buffer[j, i] = deriv(l, y[i], t[j, i])
  end
  buffer
end

@inline function value!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractMatrix, t::AbstractMatrix)
  n = size(t, 2)
  k = size(t, 1)
  @_dimcheck size(y) == size(t) && size(buffer) == (k, n)
  for i = 1:n, j = 1:k
    @inbounds buffer[j, i] = value(l, y[j, i], t[j, i])
  end
  buffer
end

@inline function grad!(buffer::AbstractMatrix, l::SupervisedLoss, y::AbstractMatrix, t::AbstractMatrix)
  n = size(t, 2)
  k = size(t, 1)
  @_dimcheck size(y) == size(t) && size(buffer) == (k, n)
  for i = 1:n, j = 1:k
    @inbounds buffer[j, i] = deriv(l, y[j, i], t[j, i])
  end
  buffer
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

# @inline value(l::MarginBasedLoss, yt::Number) = @_not_implemented
# @inline deriv(l::MarginBasedLoss, yt::Number) = @_not_implemented
# @inline deriv2(l::MarginBasedLoss, yt::Number) = @_not_implemented
# @inline value_deriv(l::MarginBasedLoss, yt::Number) = @_not_implemented

@inline function repr_fun(l::MarginBasedLoss)
  _φ(yt::Number) = value(l, yt)
  _φ
end

@inline function repr_deriv_fun(l::MarginBasedLoss)
  _φ_deriv(yt::Number) = deriv(l, yt)
  _φ_deriv
end

@inline function repr_deriv2_fun(l::MarginBasedLoss)
  _φ_deriv2(yt::Number) = deriv2(l, yt)
  _φ_deriv2
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
@inline value(l::DistanceBasedLoss, y::Number, t::Number) = value(l, y - t)
@inline deriv(l::DistanceBasedLoss, y::Number, t::Number) = deriv(l, y - t)
@inline deriv2(l::DistanceBasedLoss, y::Number, t::Number) = deriv2(l, y - t)
@inline value_deriv(l::DistanceBasedLoss, y::Number, t::Number) = value_deriv(l, y - t)

# @inline value(l::DistanceBasedLoss, r::Number) = @_not_implemented
# @inline deriv(l::DistanceBasedLoss, r::Number) = @_not_implemented
# @inline deriv2(l::DistanceBasedLoss, r::Number) = @_not_implemented
# @inline value_deriv(l::DistanceBasedLoss, r::Number) = @_not_implemented

@inline function repr_fun(l::DistanceBasedLoss)
  _ψ(r::Number) = value(l, r)
  _ψ
end

@inline function repr_deriv_fun(l::DistanceBasedLoss)
  _ψ_deriv(r::Number) = deriv(l, r)
  _ψ_deriv
end

@inline function repr_deriv2_fun(l::DistanceBasedLoss)
  _ψ_deriv2(r::Number) = deriv2(l, r)
  _ψ_deriv2
end

isdistancebased(::DistanceBasedLoss) = true
issymmetric(::DistanceBasedLoss) = false

# ==========================================================================

abstract UnsupervisedLoss <: Loss

@inline value(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = value(l, X, t)
@inline deriv(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv(l, X, t)
@inline deriv2(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv2(l, X, t)
@inline value_deriv(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = value_deriv(l, X, t)

# @inline value(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
# @inline deriv(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
# @inline deriv2(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
# @inline value_deriv(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
