
# ==========================================================================
# L(y, t) = |y - t|^P

immutable LPLoss{P} <: DistanceBasedLoss
  LPLoss() = typeof(P) <: Real ? new() : error()
end

LPLoss(p::Real) = LPLoss{p}()

value{P}(l::LPLoss{P}, r::Real) = abs(r)^P
function deriv{P}(l::LPLoss{P}, r::Real)
  if r == 0
    zero(r)
  elseif r >= 0
    P * r^(P-1)
  else
    -P * abs(r)^(P-1)
  end
end
function deriv2{P}(l::LPLoss{P}, r::Real)
  if r == 0
    zero(r)
  elseif r >= 0
    P * (P-1) * r^(P-2)
  else
    -P * (P-1) * abs(r)^(P-2)
  end
end
value_deriv{P}(l::LPLoss{P}, r::Real) = (value(l,r), deriv(l,r))

issymmetric{P}(::LPLoss{P}) = true
isdifferentiable{P}(::LPLoss{P}) = P > 1
isdifferentiable{P}(::LPLoss{P}, at) = P > 1 || at != 0
islipschitzcont{P}(::LPLoss{P}) = P == 1
isconvex{P}(::LPLoss{P}) = P >= 1

# ==========================================================================
# L(y, t) = |y - t|

typealias L1Loss LPLoss{1}

l1_loss(y::Real, t::Real) = l1_loss(y - t)
l1_loss(r::Real) = abs(r)

l1_deriv(y::Real, t::Real) = l1_deriv(y - t)
l1_deriv(r::Real) = sign(r)

l1_deriv2(y::Real, t::Real) = zero(y)
l1_deriv2(r::Real) = zero(r)

l1_loss_deriv(y::Real, t::Real) = l1_loss_deriv(y - t)
l1_loss_deriv(r::Real) = (abs(r), sign(r))

value(l::L1Loss, r::Real) = l1_loss(r)
deriv(l::L1Loss, r::Real) = l1_deriv(r)
deriv2(l::L1Loss, r::Real) = l1_deriv2(r)
value_deriv(l::L1Loss, r::Real) = l1_loss_deriv(r)

isdifferentiable(::L1Loss) = false
isdifferentiable(::L1Loss, at) = at != 0
islipschitzcont(::L1Loss) = true
isconvex(::L1Loss) = true

# ==========================================================================
# L(y, t) = (y - t)^2

typealias L2Loss LPLoss{2}

l2_loss(y::Real, t::Real) = l2_loss(y - t)
l2_loss(r::Real) = abs2(r)

l2_deriv(y::Real, t::Real) = l2_deriv(y - t)
l2_deriv(r::Real) = 2r

l2_deriv2(y::Real, t::Real) = 2
l2_deriv2(r::Real) = 2

l2_loss_deriv(y::Real, t::Real) = l2_loss_deriv(y - t)
l2_loss_deriv(r::Real) = (abs2(r), 2r)

value(l::L2Loss, r::Real) = l2_loss(r)
deriv(l::L2Loss, r::Real) = l2_deriv(r)
deriv2(l::L2Loss, r::Real) = l2_deriv2(r)
value_deriv(l::L2Loss, r::Real) = l2_loss_deriv(r)

isdifferentiable(::L2Loss) = true
isdifferentiable(::L2Loss, at) = true
islipschitzcont(::L2Loss) = false
isconvex(::L2Loss) = true
