# --------------------------------------------------------------
# convenience closures

@inline function value_fun(c::SupervisedLoss)
    _value(args...) = value(c, args...)
    _value
end

@inline function deriv_fun(c::SupervisedLoss)
    _deriv(args...) = deriv(c, args...)
    _deriv
end

@inline function deriv2_fun(c::SupervisedLoss)
    _deriv2(args...) = deriv2(c, args...)
    _deriv2
end

@inline function value_deriv_fun(c::SupervisedLoss)
    _value_deriv(args...) = value_deriv(c, args...)
    _value_deriv
end

# --------------------------------------------------------------
# These non-exported types allow for the convenience syntax
# `myloss'(y,yhat)` and `myloss''(y,yhat)` without performance
# penalty

immutable Deriv{L<:SupervisedLoss}
    loss::L
end

Base.transpose(loss::SupervisedLoss) = Deriv(loss)

(d::Deriv)(t, y) = deriv(d.loss, t, y)
(d::Deriv)(x)    = deriv(d.loss, x)

immutable Deriv2{L<:SupervisedLoss}
    loss::L
end

Base.transpose(d::Deriv) = Deriv2(d.loss)

(d::Deriv2)(t, y) = deriv2(d.loss, t, y)
(d::Deriv2)(x)    = deriv2(d.loss, x)

# --------------------------------------------------------------
# Make broadcast work for losses

Base.getindex(l::Loss, idx) = l
Base.size(::Loss) = (1,)

Base.getindex(l::Deriv, idx) = l
Base.size(::Deriv) = (1,)

Base.getindex(l::Deriv2, idx) = l
Base.size(::Deriv2) = (1,)

# --------------------------------------------------------------
# Fallback implementations

isstronglyconvex(::SupervisedLoss) = false
isminimizable(l::SupervisedLoss) = isconvex(l)
isdifferentiable(l::SupervisedLoss) = istwicedifferentiable(l)
istwicedifferentiable(::SupervisedLoss) = false
isdifferentiable(l::SupervisedLoss, at) = isdifferentiable(l)
istwicedifferentiable(l::SupervisedLoss, at) = istwicedifferentiable(l)

# Every convex function is locally lipschitz continuous
islocallylipschitzcont(l::SupervisedLoss) = isconvex(l) || islipschitzcont(l)

# If a loss if locally lipsschitz continuous then it is a
# nemitski loss
isnemitski(l::SupervisedLoss) = islocallylipschitzcont(l)
isclipable(::SupervisedLoss) = false
islipschitzcont(::SupervisedLoss) = false

ismarginbased(::SupervisedLoss) = false
isclasscalibrated(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false


# --------------------------------------------------------------
@generated function sumvalue{T,N,Q,M}(
        loss::SupervisedLoss,
        target::AbstractArray{Q,M},
        output::AbstractArray{T,N}
    )
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      val = zero(T) # TODO: this might be not be type-stable?
      @simd for I in CartesianRange(size(output))
          @nexprs $N n->(i_n = I[n])
          @inbounds val += value(loss, @nref($M,target,i), @nref($N,output,i))
      end
      val
    end
end

@generated function sumderiv{T,N,Q,M}(
        loss::SupervisedLoss,
        target::AbstractArray{Q,M},
        output::AbstractArray{T,N}
    )
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      val = zero(T) # TODO: this might be not be type-stable?
      @simd for I in CartesianRange(size(output))
          @nexprs $N n->(i_n = I[n])
          @inbounds val += deriv(loss, @nref($M,target,i), @nref($N,output,i))
      end
      val
    end
end

# --------------------------------------------------------------

@generated function meanvalue{T,N,Q,M}(
        loss::SupervisedLoss,
        target::AbstractArray{Q,M},
        output::AbstractArray{T,N}
    )
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      val = zero(T) # TODO: this might be not be type-stable?
      tmp = zero(T) # TODO: this might be not be type-stable?
      len = length(output)
      @simd for I in CartesianRange(size(output))
          @nexprs $N n->(i_n = I[n])
          @inbounds tmp = value(loss, @nref($M,target,i), @nref($N,output,i))
          tmp /= len
          val += tmp
      end
      val
    end
end

@generated function meanderiv{T,N,Q,M}(
        loss::SupervisedLoss,
        target::AbstractArray{Q,M},
        output::AbstractArray{T,N}
    )
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      val = zero(T) # TODO: this might be not be type-stable?
      tmp = zero(T) # TODO: this might be not be type-stable?
      len = length(output)
      @simd for I in CartesianRange(size(output))
          @nexprs $N n->(i_n = I[n])
          @inbounds tmp = deriv(loss, @nref($M,target,i), @nref($N,output,i))
          tmp /= len
          val += tmp
      end
      val
    end
end

# ==============================================================

# abstract MarginLoss <: SupervisedLoss

value(loss::MarginLoss, target::Number, output::Number) = value(loss, target * output)
deriv(loss::MarginLoss, target::Number, output::Number) = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)
function value_deriv(loss::MarginLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end

# TODO: consider meanvalue(loss, agreement) etc

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

# ==============================================================

# abstract DistanceLoss <: SupervisedLoss

value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)

# TODO: consider meanvalue(loss, difference) etc

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false
isclipable(::DistanceLoss) = true

