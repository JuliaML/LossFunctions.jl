
# ==========================================================================
# L(y, t) = |y - t|

immutable L1Loss <: DistanceBasedLoss end

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

issymmetric(::L1Loss) = true
isdifferentiable(::L1Loss) = true
islipschitzcont(::L1Loss) = true
isconvex(::L1Loss) = true

# ==========================================================================
# L(y, t) = (y - t)^2

immutable L2Loss <: DistanceBasedLoss end

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

issymmetric(::L2Loss) = true
isdifferentiable(::L2Loss) = true
islipschitzcont(::L2Loss) = false
isconvex(::L2Loss) = true
