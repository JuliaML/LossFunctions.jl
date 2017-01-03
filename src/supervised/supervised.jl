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
Base.size(::Loss) = ()

Base.getindex(l::Deriv, idx) = l
Base.size(::Deriv) = ()

Base.getindex(l::Deriv2, idx) = l
Base.size(::Deriv2) = ()

# --------------------------------------------------------------
# Fallback implementations

function value_deriv(l::SupervisedLoss, target::Number, output::Number)
    value(l, target, output), deriv(l, target, output)
end

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

for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # By default compute the element-wise result
        function ($FUN)(loss::SupervisedLoss,
                        target::AbstractArray,
                        output::AbstractArray)
            ($FUN)(loss, target, output, AvgMode.None())
        end

        # Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        function ($FUN){T,Q,N}(loss::SupervisedLoss,
                               target::AbstractArray{Q,N},
                               output::AbstractArray{T,N},
                               avg::AverageMode,
                               ::ObsDim.Last)
            ($FUN)(loss, target, output, avg, ObsDim.Constant{N}())
        end

        # Compute element-wise (returns an array)
        @generated function ($FUN){Q,M,T,N}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.None)
            quote
                S = typeof(($($FUN))(loss, one(Q), one(T)))
                ($($FUN)).(loss, target, output)::Array{S,$(max(N,M))}
            end
        end

        # Compute the sum per observation (returns a Vector)
        function ($FUN){Q,T,N,O}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AvgMode.Sum,
                ::ObsDim.Constant{O})
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            k = prod(size(output,n) for n in 1:N if n != O)
            S = typeof(zero(($FUN)(loss, one(Q), one(T))))
            out = zeros(S, size(output, O))
            @inbounds @simd for I in CartesianRange(size(output))
                out[I[O]] += ($FUN)(loss, target[I], output[I])
            end
            out
        end

        # Compute the sum (returns a Number)
        @generated function ($FUN){Q,M,T,N}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.Sum)
            M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
            quote
                @nexprs $M (n)->@_dimcheck(size(target, n) == size(output, n))
                out = zero(($($FUN))(loss, one(Q), one(T)))
                @inbounds @simd for I in CartesianRange(size(output))
                    @nexprs $N n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i))
                end
                out
            end
        end

        # Compute the average per observation (returns a Vector)
        function ($FUN){Q,T,N,O}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AvgMode.Mean,
                ::ObsDim.Constant{O})
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            k = prod(size(output,n) for n in 1:N if n != O)
            S = typeof(zero(($FUN)(loss, one(Q), one(T))) / k)
            out = zeros(S, size(output, O))
            @inbounds @simd for I in CartesianRange(size(output))
                out[I[O]] = ($FUN)(loss, target[I], output[I]) / k
            end
            out
        end

        # Compute the total average (returns a Number)
        @generated function ($FUN){Q,M,T,N}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.Mean)
            M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
            quote
                @nexprs $M (n)->@_dimcheck(size(target, n) == size(output, n))
                len = length(output)
                out = zero(($($FUN))(loss, one(Q), one(T))) / len
                @inbounds @simd for I in CartesianRange(size(output))
                    @nexprs $N n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i)) / len
                end
                out
            end
        end

        # Compute the total weighted average (returns a Number)
        function ($FUN){Q,T,N,O}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AvgMode.WeightedMean,
                ::ObsDim.Constant{O})
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            @_dimcheck size(output, O) == length(avg.weights)
            k = prod(size(output,n) for n in 1:N if n != O)
            nrm = avg.normalize ? k * sum(avg.weights) : k * one(sum(avg.weights))
            out = zero(($FUN)(loss, one(Q), one(T)) * (avg.weights[1] / nrm))
            @inbounds @simd for I in CartesianRange(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (avg.weights[I[O]] / nrm)
            end
            out
        end

        # Compute the total weighted sum (returns a Number)
        function ($FUN){Q,T,N,O}(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AvgMode.WeightedSum,
                ::ObsDim.Constant{O})
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            @_dimcheck size(output, O) == length(avg.weights)
            nrm = avg.normalize ? sum(avg.weights) : one(sum(avg.weights))
            out = zero(($FUN)(loss, one(Q), one(T)) * (avg.weights[1] / nrm))
            @inbounds @simd for I in CartesianRange(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (avg.weights[I[O]] / nrm)
            end
            out
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin
            # By default compute the element-wise result
            function ($FUN)(loss::$KIND, numbers::AbstractArray)
                ($FUN)(loss, numbers, AvgMode.None())
            end

            # Compute element-wise (returns an array)
            function ($FUN){T,N}(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.None)
                S = typeof(($FUN)(loss, one(T)))
                ($FUN).(loss, numbers)::Array{S,N}
            end

            # Compute the sum (returns a Number)
            function ($FUN){T,N}(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.Sum)
                S = typeof(($FUN)(loss, one(T)))
                reduce(+, zero(S),
                       (($FUN)(loss, num) for num in numbers))
            end

            # Compute the mean (returns a Number)
            function ($FUN){T,N}(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.Mean)
                len = length(numbers)
                S = typeof(($FUN)(loss, one(T)) / len)
                reduce(+, zero(S),
                       (($FUN)(loss, num) / len for num in numbers))
            end
        end
    end
end

# --------------------------------------------------------------

# abstract MarginLoss <: SupervisedLoss

value(loss::MarginLoss, target::Number, output::Number) = value(loss, target * output)
deriv(loss::MarginLoss, target::Number, output::Number) = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)
function value_deriv(loss::MarginLoss, target::Number, output::Number)
    v, d = value_deriv(loss, target * output)
    (v, target*d)
end

# Fallback for losses that don't want to take advantage of this
value_deriv(loss::MarginLoss, agreement::Number) = (value(loss, agreement), deriv(loss, agreement))

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

# --------------------------------------------------------------

# abstract DistanceLoss <: SupervisedLoss

value(loss::DistanceLoss, target::Number, output::Number) = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number) = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)
value_deriv(loss::DistanceLoss, target::Number, output::Number) = value_deriv(loss, output - target)

# Fallback for losses that don't want to take advantage of this
value_deriv(loss::DistanceLoss, difference::Number) = (value(loss, difference), deriv(loss, difference))

isdistancebased(::DistanceLoss) = true
issymmetric(::DistanceLoss) = false
isclipable(::DistanceLoss) = true

