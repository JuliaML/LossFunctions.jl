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
# non-exported types

struct Deriv{L<:SupervisedLoss}
    loss::L
end
@inline (d::Deriv)(args...) = deriv(d.loss, args...)

struct Deriv2{L<:SupervisedLoss}
    loss::L
end
@inline (d::Deriv2)(args...) = deriv2(d.loss, args...)

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

        # ------------------------------------------------------
        # FALLBACK

        # By default compute the element-wise result
        @inline function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray)
            ($FUN)(loss, target, output, AvgMode.None())
        end

        # (mutating) By default compute the element-wise result
        @inline function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray)
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, AvgMode.None())
        end

        # Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        @inline function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray{T,N},
                avg::AverageMode,
                ::ObsDim.Last = ObsDim.Last()) where {T,N}
            ($FUN)(loss, target, output, avg, ObsDim.Constant{N}())
        end

        # (mutating) Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        @inline function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray,
                output::AbstractArray{T,N},
                avg::AverageMode,
                ::ObsDim.Last = ObsDim.Last()) where {T,N}
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, ObsDim.Constant{N}())
        end

        # ------------------------------------------------------
        # ELEMENT-WISE BROADCAST

        # Compute element-wise (returns an array)
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.None) where {Q,M,T,N}
            quote
                $(Expr(:meta, :inline))
                S = typeof(($($FUN))(loss, one(Q), one(T)))
                ($($FUN)).(loss, target, output)::Array{S,$(max(N,M))}
            end
        end

        # (mutating) Compute element-wise (returns an array)
        function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.None) where {Q,M,T,N}
            buffer .= ($FUN).(loss, target, output)
            buffer
        end

        # ------------------------------------------------------
        # REDUCE TO NUMBER - BROADCAST

        # Compute the mean (returns a Number)
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.Mean) where {Q,M,T,N}
            bigger = M > N ? :target : :output
            S, B = min(M,N), max(M,N)
            P = promote_type(Q,T)
            quote
                @nexprs $S (n)->@_dimcheck(size(target, n) == size(output, n))
                nrm = 1 / $P(length($bigger))
                out = zero(($($FUN))(loss, one(Q), one(T)) * nrm)
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i)) * nrm
                end
                out
            end
        end

        # Compute the sum (returns a Number)
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AvgMode.Sum) where {Q,M,T,N}
            bigger = M > N ? :target : :output
            S, B = min(M,N), max(M,N)
            quote
                @nexprs $S (n)->@_dimcheck(size(target, n) == size(output, n))
                out = zero(($($FUN))(loss, one(Q), one(T)))
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i))
                end
                out
            end
        end

        # ------------------------------------------------------
        # REDUCE TO NUMBER - SAME SIZE

        # Compute the total weighted mean (returns a Number)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AvgMode.WeightedMean,
                ::ObsDim.Constant{O}) where {Q,T,N,O}
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            @_dimcheck size(output, O) == length(avg.weights)
            k = prod(n != O ? size(output,n) : 1 for n in 1:N)::Int
            nrm = avg.normalize ? inv(k * sum(avg.weights)) : inv(k * one(sum(avg.weights)))
            out = zero(($FUN)(loss, one(Q), one(T)) * (avg.weights[1] * nrm))
            @inbounds @simd for I in CartesianIndices(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (avg.weights[I[O]] * nrm)
            end
            out
        end

        # Compute the total weighted sum (returns a Number)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AvgMode.WeightedSum,
                ::ObsDim.Constant{O}) where {Q,T,N,O}
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            @_dimcheck size(output, O) == length(avg.weights)
            nrm = avg.normalize ? inv(sum(avg.weights)) : inv(one(sum(avg.weights)))
            out = zero(($FUN)(loss, one(Q), one(T)) * (avg.weights[1] * nrm))
            @inbounds @simd for I in CartesianIndices(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (avg.weights[I[O]] * nrm)
            end
            out
        end

        # ------------------------------------------------------
        # PER OBSERVATION - SAME SIZE - TO VECTOR

        # Compute the mean per observation (returns a Vector)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AvgMode.Mean,
                obsdim::ObsDim.Constant{O}) where {Q,T,N,O}
            S = typeof(($FUN)(loss, one(Q), one(T)) / one(Int))
            buffer = zeros(S, size(output, O))
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, obsdim)
        end

        # (mutating) Compute the mean per observation (returns a Vector)
        function ($(Symbol(FUN,:!)))(
                buffer::AbstractVector{B},
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AvgMode.Mean,
                ::ObsDim.Constant{O}) where {B,Q,T,N,O}
            N == 1 && throw(ArgumentError("Mean per observation non sensible for two Vectors. Try omitting the obsdim"))
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            @_dimcheck length(buffer) == size(output, O)
            fill!(buffer, zero(B))
            P = promote_type(Q,T)
            k = P(prod(size(output,n) for n in 1:N if n != O))
            nrm = 1 / k
            @inbounds @simd for I in CartesianIndices(size(output))
                buffer[I[O]] += ($FUN)(loss, target[I], output[I]) * nrm
            end
            buffer
        end

        # Compute the sum per observation (returns a Vector)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AvgMode.Sum,
                obsdim::ObsDim.Constant{O}) where {Q,T,N,O}
            S = typeof(($FUN)(loss, one(Q), one(T)))
            buffer = zeros(S, size(output, O))
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, obsdim)
        end

        # (mutating) Compute the sum per observation (returns a Vector)
        function ($(Symbol(FUN,:!)))(
                buffer::AbstractVector{B},
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AvgMode.Sum,
                ::ObsDim.Constant{O}) where {B,Q,T,N,O}
            N == 1 && throw(ArgumentError("Sum per observation non sensible for two Vectors. Try omitting the obsdim"))
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @_dimcheck size(target) == size(output)
            @_dimcheck length(buffer) == size(output, O)
            fill!(buffer, zero(B))
            @inbounds @simd for I in CartesianIndices(size(output))
                buffer[I[O]] += ($FUN)(loss, target[I], output[I])
            end
            buffer
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin

            # --------------------------------------------------
            # FALLBACK

            # By default compute the element-wise result
            @inline function ($FUN)(loss::$KIND, numbers::AbstractArray)
                ($FUN)(loss, numbers, AvgMode.None())
            end

            # (mutating) By default compute the element-wise result
            @inline function ($(Symbol(FUN,:!)))(
                    buffer::AbstractArray,
                    loss::$KIND,
                    numbers::AbstractArray)
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, AvgMode.None())
            end

            # Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
            @inline function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AverageMode,
                    ::ObsDim.Last = ObsDim.Last()) where {T,N}
                ($FUN)(loss, numbers, avg, ObsDim.Constant{N}())
            end

            # (mutating) Translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
            @inline function ($(Symbol(FUN,:!)))(
                    buffer::AbstractArray,
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AverageMode,
                    ::ObsDim.Last = ObsDim.Last()) where {T,N}
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, avg, ObsDim.Constant{N}())
            end

            # --------------------------------------------------
            # ELEMENT-WISE BROADCAST

            # Compute element-wise (returns an array)
            @inline function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.None) where {T,N}
                S = typeof(($FUN)(loss, one(T)))
                ($FUN).(loss, numbers)::Array{S,N}
            end

            # (mutating) Compute element-wise (returns an array)
            function ($(Symbol(FUN,:!)))(
                    buffer::AbstractArray,
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.None) where {T,N}
                buffer .= ($FUN).(loss, numbers)
                buffer
            end

            # --------------------------------------------------
            # REDUCE TO NUMBER

            # Compute the mean (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.Mean) where {T,N}
                nrm = 1 / length(numbers)
                S = typeof(($FUN)(loss, one(T)) * one(nrm))
                mapreduce(x -> loss(x) * nrm, +, numbers)::S
            end

            # Compute the sum (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.Sum) where {T,N}
                mapreduce(loss, +, numbers)
            end

            # Compute the total weighted mean (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AvgMode.WeightedMean,
                    ::ObsDim.Constant{O}) where {T,N,O}
                @_dimcheck size(numbers, O) == length(avg.weights)
                k = prod(n != O ? size(numbers,n) : 1 for n in 1:N)::Int
                nrm = avg.normalize ? inv(k * sum(avg.weights)) : inv(k * one(sum(avg.weights)))
                out = zero(($FUN)(loss, one(T)) * (avg.weights[1] * nrm))
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    out += ($FUN)(loss, numbers[I]) * (avg.weights[I[O]] * nrm)
                end
                out
            end

            # Compute the total weighted sum (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AvgMode.WeightedSum,
                    ::ObsDim.Constant{O}) where {T,N,O}
                @_dimcheck size(numbers, O) == length(avg.weights)
                nrm = avg.normalize ? inv(sum(avg.weights)) : inv(one(sum(avg.weights)))
                out = zero(($FUN)(loss, one(T)) * (avg.weights[1] * nrm))
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    out += ($FUN)(loss, numbers[I]) * (avg.weights[I[O]] * nrm)
                end
                out
            end

            # --------------------------------------------------
            # PER OBSERVATION - SAME SIZE - TO VECTOR

            # Compute the mean per observation (returns a Vector)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AvgMode.Mean,
                    obsdim::ObsDim.Constant{O}) where {T,N,O}
                S = typeof(($FUN)(loss, one(T)) / one(Int))
                buffer = zeros(S, size(numbers, O))
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, avg, obsdim)
            end

            # (mutating) Compute the mean per observation (returns a Vector)
            function ($(Symbol(FUN,:!)))(
                    buffer::AbstractVector{B},
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.Mean,
                    ::ObsDim.Constant{O}) where {B,T,N,O}
                N == 1 && throw(ArgumentError("Mean per observation non sensible for two Vectors. Try omitting the obsdim"))
                O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
                @_dimcheck length(buffer) == size(numbers, O)
                fill!(buffer, zero(B))
                k = prod(size(numbers,n) for n in 1:N if n != O)::Int
                nrm = 1 / k
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    buffer[I[O]] += ($FUN)(loss, numbers[I]) * nrm
                end
                buffer
            end

            # Compute the sum per observation (returns a Vector)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AvgMode.Sum,
                    obsdim::ObsDim.Constant{O}) where {T,N,O}
                S = typeof(($FUN)(loss, one(T)))
                buffer = zeros(S, size(numbers, O))
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, avg, obsdim)
            end

            # (mutating) Compute the sum per observation (returns a Vector)
            function ($(Symbol(FUN,:!)))(
                    buffer::AbstractVector{B},
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AvgMode.Sum,
                    ::ObsDim.Constant{O}) where {B,T,N,O}
                N == 1 && throw(ArgumentError("Sum per observation non sensible for two Vectors. Try omitting the obsdim"))
                O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
                @_dimcheck length(buffer) == size(numbers, O)
                fill!(buffer, zero(B))
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    buffer[I[O]] += ($FUN)(loss, numbers[I])
                end
                buffer
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
