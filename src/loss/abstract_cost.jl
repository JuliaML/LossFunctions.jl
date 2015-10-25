abstract Cost

call(c::Cost, X::AbstractArray, y::Number, α::Number) = value(c, X, y, α)
value(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented
deriv(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented
deriv2(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented
value_deriv(c::Cost, X::AbstractArray, y::Number, α::Number) = @_not_implemented

function value_fun(c::Cost)
  _value(args...) = value(c, args...)
  _value
end

function deriv_fun(c::Cost)
  _deriv(args...) = deriv(c, args...)
  _deriv
end

function deriv2_fun(c::Cost)
  _deriv2(args...) = deriv2(c, args...)
  _deriv2
end

function value_deriv_fun(c::Cost)
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

value(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = value(l, y, t)
deriv(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv(l, y, t)
deriv2(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv2(l, y, t)
value_deriv(l::SupervisedLoss, X::AbstractArray, y::Number, t::Number) = value_deriv(l, y, t)

call(l::SupervisedLoss, y::Number, t::Number) = value(l, y, t)
value(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented
deriv(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented
deriv2(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented
value_deriv(l::SupervisedLoss, y::Number, t::Number) = @_not_implemented

ismarginbased(::SupervisedLoss) = false
isclasscalibrated(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false

# ==========================================================================

abstract MarginBasedLoss <: SupervisedLoss

value(l::MarginBasedLoss, y::Number, t::Number) = value(l, y * t)
deriv(l::MarginBasedLoss, y::Number, t::Number) = deriv(l, y * t)
deriv2(l::MarginBasedLoss, y::Number, t::Number) = deriv2(l, y * t)
value_deriv(l::MarginBasedLoss, y::Number, t::Number) = value_deriv(l, y * t)

call(l::MarginBasedLoss, yt::Number) = value(l, yt)
transpose(l::MarginBasedLoss) = repr_deriv_fun(l)
value(l::MarginBasedLoss, yt::Number) = @_not_implemented
deriv(l::MarginBasedLoss, yt::Number) = @_not_implemented
deriv2(l::MarginBasedLoss, yt::Number) = @_not_implemented
value_deriv(l::MarginBasedLoss, yt::Number) = @_not_implemented

function repr_fun(l::MarginBasedLoss)
  _φ(yt::Number) = value(l, yt)
  _φ
end

function repr_deriv_fun(l::MarginBasedLoss)
  _φ_deriv(yt::Number) = deriv(l, yt)
  _φ_deriv
end

function repr_deriv2_fun(l::MarginBasedLoss)
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

value(l::DistanceBasedLoss, y::Number, t::Number) = value(l, y - t)
deriv(l::DistanceBasedLoss, y::Number, t::Number) = deriv(l, y - t)
deriv2(l::DistanceBasedLoss, y::Number, t::Number) = deriv2(l, y - t)
value_deriv(l::DistanceBasedLoss, y::Number, t::Number) = value_deriv(l, y - t)

call(l::DistanceBasedLoss, r::Number) = value(l, r)
transpose(l::DistanceBasedLoss) = repr_deriv_fun(l)
value(l::DistanceBasedLoss, r::Number) = @_not_implemented
deriv(l::DistanceBasedLoss, r::Number) = @_not_implemented
deriv2(l::DistanceBasedLoss, r::Number) = @_not_implemented
value_deriv(l::DistanceBasedLoss, r::Number) = @_not_implemented

function repr_fun(l::DistanceBasedLoss)
  _ψ(r::Number) = value(l, r)
  _ψ
end

function repr_deriv_fun(l::DistanceBasedLoss)
  _ψ_deriv(r::Number) = deriv(l, r)
  _ψ_deriv
end

function repr_deriv2_fun(l::DistanceBasedLoss)
  _ψ_deriv2(r::Number) = deriv2(l, r)
  _ψ_deriv2
end

isdistancebased(::DistanceBasedLoss) = true
issymmetric(::DistanceBasedLoss) = false

# ==========================================================================

abstract UnsupervisedLoss <: Loss

value(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = value(l, X, t)
deriv(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv(l, X, t)
deriv2(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = deriv2(l, X, t)
value_deriv(l::UnsupervisedLoss, X::AbstractArray, y::Number, t::Number) = value_deriv(l, X, t)

call(l::UnsupervisedLoss, X::AbstractArray, t::Number) = value(l, X, t)
value(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
deriv(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
deriv2(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
value_deriv(l::UnsupervisedLoss, X::AbstractArray, t::Number) = @_not_implemented
