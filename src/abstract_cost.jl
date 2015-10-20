abstract Cost

call(c::Cost, X::AbstractArray, y::Real, α::Real) = value(c, X, y, α)
value(c::Cost, X::AbstractArray, y::Real, α::Real) = @_not_implemented
deriv(c::Cost, X::AbstractArray, y::Real, α::Real) = @_not_implemented
deriv2(c::Cost, X::AbstractArray, y::Real, α::Real) = @_not_implemented
value_deriv(c::Cost, X::AbstractArray, y::Real, α::Real) = @_not_implemented

function value_fun(c::Cost)
  _value(X::AbstractArray, y::Real, α::Real) = value(c, X, y, α)
  _value
end

function deriv_fun(c::Cost)
  _deriv(X::AbstractArray, y::Real, α::Real) = deriv(c, X, y, α)
  _deriv
end

function deriv2_fun(c::Cost)
  _deriv2(X::AbstractArray, y::Real, α::Real) = deriv2(c, X, y, α)
  _deriv2
end

function value_deriv_fun(c::Cost)
  _value_deriv(X::AbstractArray, y::Real, α::Real) = value_deriv(c, X, y, α)
  _value_deriv
end

isminimizable(c::Cost) = isconvex(c)
isdifferentiable(::Cost) = false
isdifferentiable(c::Cost, at) = isdifferentiable(c)
isconvex(::Cost) = false

# ==========================================================================

abstract Loss <: Cost

isnemitski(l::Loss) = islocallylipschitzcont(l)
islipschitzcont(::Loss) = false
islocallylipschitzcont(::Loss) = false
isclipable(::Loss) = false
isclipable(l::Loss, M::Real) = M > 0 ? isclipable(l) : false

# ==========================================================================

abstract SupervisedLoss <: Loss

value(l::SupervisedLoss, X::AbstractArray, y::Real, t::Real) = value(l, y, t)
deriv(l::SupervisedLoss, X::AbstractArray, y::Real, t::Real) = deriv(l, y, t)
deriv2(l::SupervisedLoss, X::AbstractArray, y::Real, t::Real) = deriv2(l, y, t)
value_deriv(l::SupervisedLoss, X::AbstractArray, y::Real, t::Real) = value_deriv(l, y, t)

call(l::SupervisedLoss, y::Real, t::Real) = value(l, y, t)
value(l::SupervisedLoss, y::Real, t::Real) = @_not_implemented
deriv(l::SupervisedLoss, y::Real, t::Real) = @_not_implemented
deriv2(l::SupervisedLoss, y::Real, t::Real) = @_not_implemented
value_deriv(l::SupervisedLoss, y::Real, t::Real) = @_not_implemented

function value_fun(c::SupervisedLoss)
  _value(y::Real, t::Real) = value(c, y, t)
  _value
end

function deriv_fun(c::SupervisedLoss)
  _deriv(y::Real, t::Real) = deriv(c, y, t)
  _deriv
end

function deriv2_fun(c::SupervisedLoss)
  _deriv2(y::Real, t::Real) = deriv2(c, y, t)
  _deriv2
end

function value_deriv_fun(c::SupervisedLoss)
  _value_deriv(y::Real, t::Real) = value_and_deriv(c, y, t)
  _value_deriv
end

ismarginbased(::SupervisedLoss) = false
isclasscalibrated(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false

# ==========================================================================

abstract MarginBasedLoss <: SupervisedLoss

value(l::MarginBasedLoss, y::Real, t::Real) = value(l, y * t)
deriv(l::MarginBasedLoss, y::Real, t::Real) = deriv(l, y * t)
deriv2(l::MarginBasedLoss, y::Real, t::Real) = deriv2(l, y * t)
value_deriv(l::MarginBasedLoss, y::Real, t::Real) = value_deriv(l, y * t)

call(l::MarginBasedLoss, yt::Real) = value(l, yt)
transpose(l::MarginBasedLoss) = representing_deriv_fun(l)
value(l::MarginBasedLoss, yt::Real) = @_not_implemented
deriv(l::MarginBasedLoss, yt::Real) = @_not_implemented
deriv2(l::MarginBasedLoss, yt::Real) = @_not_implemented
value_deriv(l::MarginBasedLoss, yt::Real) = @_not_implemented

function repr_fun(l::MarginBasedLoss)
  _φ(yt::Real) = value(l, yt)
  _φ
end

function repr_deriv_fun(l::MarginBasedLoss)
  _φ_deriv(yt::Real) = deriv(l, yt)
  _φ_deriv
end

function repr_deriv2_fun(l::MarginBasedLoss)
  _φ_deriv2(yt::Real) = deriv2(l, yt)
  _φ_deriv2
end

isunivfishercons(::MarginBasedLoss) = false
isfishercons(l::MarginBasedLoss) = isunivfishercons(l)
isnemitski(::MarginBasedLoss) = true
islocallylipschitzcont(l::MarginBasedLoss) = isconvex(l)
ismarginbased(::MarginBasedLoss) = true
isclasscalibrated(l::MarginBasedLoss) = isconvex(l) && isdifferentiable(l) && deriv(l, 0) < 0

# ==========================================================================

abstract DistanceBasedLoss <: SupervisedLoss

value(l::DistanceBasedLoss, y::Real, t::Real) = value(l, y - t)
deriv(l::DistanceBasedLoss, y::Real, t::Real) = deriv(l, y - t)
deriv2(l::DistanceBasedLoss, y::Real, t::Real) = deriv2(l, y - t)
value_deriv(l::DistanceBasedLoss, y::Real, t::Real) = value_deriv(l, y - t)

call(l::DistanceBasedLoss, r::Real) = value(l, r)
transpose(l::DistanceBasedLoss) = representing_deriv_fun(l)
value(l::DistanceBasedLoss, r::Real) = @_not_implemented
deriv(l::DistanceBasedLoss, r::Real) = @_not_implemented
deriv2(l::DistanceBasedLoss, r::Real) = @_not_implemented
value_deriv(l::DistanceBasedLoss, r::Real) = @_not_implemented

function repr_fun(l::DistanceBasedLoss)
  _ψ(r::Real) = value(l, r)
  _ψ
end

function repr_deriv_fun(l::DistanceBasedLoss)
  _ψ_deriv(r::Real) = deriv(l, r)
  _ψ_deriv
end

function repr_deriv2_fun(l::DistanceBasedLoss)
  _ψ_deriv2(r::Real) = deriv2(l, r)
  _ψ_deriv2
end

isdistancebased(::DistanceBasedLoss) = true
issymmetric(::DistanceBasedLoss) = false

# ==========================================================================

abstract UnsupervisedLoss <: Loss

value(l::UnsupervisedLoss, X::AbstractArray, y::Real, t::Real) = value(l, X, t)
deriv(l::UnsupervisedLoss, X::AbstractArray, y::Real, t::Real) = deriv(l, X, t)
deriv2(l::UnsupervisedLoss, X::AbstractArray, y::Real, t::Real) = deriv2(l, X, t)
value_deriv(l::UnsupervisedLoss, X::AbstractArray, y::Real, t::Real) = value_deriv(l, X, t)

call(l::UnsupervisedLoss, X::AbstractArray, t::Real) = value(l, X, t)
value(l::UnsupervisedLoss, X::AbstractArray, t::Real) = @_not_implemented
deriv(l::UnsupervisedLoss, X::AbstractArray, t::Real) = @_not_implemented
deriv2(l::UnsupervisedLoss, X::AbstractArray, t::Real) = @_not_implemented
value_deriv(l::UnsupervisedLoss, X::AbstractArray, t::Real) = @_not_implemented
