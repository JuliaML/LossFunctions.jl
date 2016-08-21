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

isminimizable(c::SupervisedLoss) = isconvex(c)
isdifferentiable(c::SupervisedLoss) = istwicedifferentiable(c)
istwicedifferentiable(::SupervisedLoss) = false
isdifferentiable(c::SupervisedLoss, at) = isdifferentiable(c)
istwicedifferentiable(c::SupervisedLoss, at) = istwicedifferentiable(c)
isconvex(::SupervisedLoss) = false
isstronglyconvex(::SupervisedLoss) = false

isnemitski(loss::SupervisedLoss) = islocallylipschitzcont(loss)
islipschitzcont(::SupervisedLoss) = false
islocallylipschitzcont(::SupervisedLoss) = false
isclipable(::SupervisedLoss) = false
islipschitzcont_deriv(::SupervisedLoss) = false

ismarginbased(::SupervisedLoss) = false
isclasscalibrated(::SupervisedLoss) = false
isdistancebased(::SupervisedLoss) = false
issymmetric(::SupervisedLoss) = false

# --------------------------------------------------------------

@inline function value(loss::SupervisedLoss, target::AbstractArray, output::AbstractArray)
    buffer = similar(output)
    value!(buffer, loss, target, output)
end

@inline function deriv(loss::SupervisedLoss, target::AbstractArray, output::AbstractArray)
    buffer = similar(output)
    deriv!(buffer, loss, target, output)
end

# TODO: same for deriv2

# TODO: same for value_deriv

# --------------------------------------------------------------
# value!, deriv!
# `output` can have more dimensions than `target`, in which case do broadcasting

@generated function value!{T,N,Q,M}(
        buffer::AbstractArray,
        loss::SupervisedLoss,
        target::AbstractArray{Q,M},
        output::AbstractArray{T,N}
    )
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @_dimcheck size(buffer) == size(output)
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      @simd for I in CartesianRange(size(output))
          @nexprs $N n->(i_n = I[n])
          @inbounds @nref($N,buffer,i) = value(loss, @nref($M,target,i), @nref($N,output,i))
      end
      buffer
    end
end

@generated function deriv!{T,N,Q,M}(
        buffer::AbstractArray,
        loss::SupervisedLoss,
        target::AbstractArray{Q,M},
        output::AbstractArray{T,N}
    )
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @_dimcheck size(buffer) == size(output)
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      @simd for I in CartesianRange(size(output))
          @nexprs $N n->(i_n = I[n])
          @inbounds @nref($N,buffer,i) = deriv(loss, @nref($M,target,i), @nref($N,output,i))
      end
      buffer
    end
end

# TODO: same for deriv2

# TODO: same for value_deriv

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

isunivfishercons(::MarginLoss) = false
isfishercons(loss::MarginLoss) = isunivfishercons(loss)
isnemitski(::MarginLoss) = true
islocallylipschitzcont(loss::MarginLoss) = isconvex(loss)
ismarginbased(::MarginLoss) = true
isclasscalibrated(loss::MarginLoss) = isconvex(loss) && isdifferentiable(loss, 0) && deriv(loss, 0) < 0

# ==============================================================

# abstract DistanceLoss <: SupervisedLoss

value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false

