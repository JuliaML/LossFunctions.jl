# broadcasting behavior
Broadcast.broadcastable(loss::SupervisedLoss) = Ref(loss)

# fallback to unary evaluation
value(loss::DistanceLoss, target::Number, output::Number)  = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number)  = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)

value(loss::MarginLoss, target::Number, output::Number)  = value(loss, target * output)
deriv(loss::MarginLoss, target::Number, output::Number)  = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)

# ------------------
# AVAILABLE LOSSES
# ------------------
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/other.jl")

# meta-losses
include("supervised/ordinal.jl")
include("supervised/scaled.jl")
include("supervised/weighted.jl")

# ------------------------------
# DEFAULT AGGREGATION BEHAVIOR
# ------------------------------
for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # by default compute the element-wise result
        @inline function ($FUN)(
                loss::SupervisedLoss,
                targets::AbstractArray,
                outputs::AbstractArray)
            ($FUN)(loss, targets, outputs, AggMode.None())
        end

        # (mutating) by default compute the element-wise result
        @inline function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                targets::AbstractArray,
                outputs::AbstractArray)
            ($(Symbol(FUN,:!)))(buffer, loss, targets, outputs, AggMode.None())
        end

        # translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        @inline function ($FUN)(
                loss::SupervisedLoss,
                targets::AbstractArray,
                outputs::AbstractArray{T,N},
                agg::AggregateMode,
                ::ObsDim.Last = ObsDim.Last()) where {T,N}
            ($FUN)(loss, targets, outputs, agg, ObsDim.Constant{N}())
        end

        # (mutating) translate ObsDim.Last to the correct ObsDim.Constant (for code reduction)
        @inline function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                targets::AbstractArray,
                outputs::AbstractArray{T,N},
                agg::AggregateMode,
                ::ObsDim.Last = ObsDim.Last()) where {T,N}
            ($(Symbol(FUN,:!)))(buffer, loss, targets, outputs, agg, ObsDim.Constant{N}())
        end

        # -------------------
        # AGGREGATION: NONE
        # -------------------
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.None) where {Q,M,T,N}
            quote
                $(Expr(:meta, :inline))
                S = typeof(($($FUN))(loss, one(Q), one(T)))
                ($($FUN)).(Ref(loss), target, output)
            end
        end

        function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.None) where {Q,M,T,N}
            buffer .= ($FUN).(Ref(loss), target, output)
            buffer
        end

        # ------------------
        # AGGREGATION: SUM
        # ------------------
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.Sum) where {Q,M,T,N}
            bigger = M > N ? :target : :output
            S, B = min(M,N), max(M,N)
            quote
                @nexprs $S (n)->@dimcheck(size(target, n) == size(output, n))
                out = zero(($($FUN))(loss, one(Q), one(T)))
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i))
                end
                out
            end
        end

        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AggMode.Sum,
                obsdim::ObsDim.Constant{O}) where {Q,T,N,O}
            S = typeof(($FUN)(loss, one(Q), one(T)))
            buffer = zeros(S, size(output, O))
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, obsdim)
        end

        function ($(Symbol(FUN,:!)))(
                buffer::AbstractVector{B},
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AggMode.Sum,
                ::ObsDim.Constant{O}) where {B,Q,T,N,O}
            N == 1 && throw(ArgumentError("Sum per observation non sensible for two Vectors. Try omitting the obsdim"))
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck length(buffer) == size(output, O)
            fill!(buffer, zero(B))
            @inbounds @simd for I in CartesianIndices(size(output))
                buffer[I[O]] += ($FUN)(loss, target[I], output[I])
            end
            buffer
        end

        # -------------------
        # AGGREGATION: MEAN
        # -------------------
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.Mean) where {Q,M,T,N}
            bigger = M > N ? :target : :output
            S, B = min(M,N), max(M,N)
            P = promote_type(Q,T)
            quote
                @nexprs $S (n)->@dimcheck(size(target, n) == size(output, n))
                nrm = 1 / $P(length($bigger))
                out = zero(($($FUN))(loss, one(Q), one(T)) * nrm)
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i)) * nrm
                end
                out
            end
        end

        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AggMode.Mean,
                obsdim::ObsDim.Constant{O}) where {Q,T,N,O}
            S = typeof(($FUN)(loss, one(Q), one(T)) / one(Int))
            buffer = zeros(S, size(output, O))
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, obsdim)
        end

        function ($(Symbol(FUN,:!)))(
                buffer::AbstractVector{B},
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AggMode.Mean,
                ::ObsDim.Constant{O}) where {B,Q,T,N,O}
            N == 1 && throw(ArgumentError("Mean per observation non sensible for two Vectors. Try omitting the obsdim"))
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck length(buffer) == size(output, O)
            fill!(buffer, zero(B))
            P = promote_type(Q,T)
            k = P(prod(size(output,n) for n in 1:N if n != O))
            nrm = 1 / k
            @inbounds @simd for I in CartesianIndices(size(output))
                buffer[I[O]] += ($FUN)(loss, target[I], output[I]) * nrm
            end
            buffer
        end

        # ---------------------------
        # AGGREGATION: WEIGHTED SUM
        # ---------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                agg::AggMode.WeightedSum,
                ::ObsDim.Constant{O}) where {Q,T,N,O}
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck size(output, O) == length(agg.weights)
            nrm = agg.normalize ? inv(sum(agg.weights)) : inv(one(sum(agg.weights)))
            out = zero(($FUN)(loss, one(Q), one(T)) * (agg.weights[1] * nrm))
            @inbounds @simd for I in CartesianIndices(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (agg.weights[I[O]] * nrm)
            end
            out
        end

        # ----------------------------
        # AGGREGATION: WEIGHTED MEAN
        # ----------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                agg::AggMode.WeightedMean,
                ::ObsDim.Constant{O}) where {Q,T,N,O}
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck size(output, O) == length(agg.weights)
            k = prod(n != O ? size(output,n) : 1 for n in 1:N)::Int
            nrm = agg.normalize ? inv(k * sum(agg.weights)) : inv(k * one(sum(agg.weights)))
            out = zero(($FUN)(loss, one(Q), one(T)) * (agg.weights[1] * nrm))
            @inbounds @simd for I in CartesianIndices(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (agg.weights[I[O]] * nrm)
            end
            out
        end
    end
end

# TODO: find way to use normal broadcast for this
# probably with the new changes in MLMetric and compare modes
include("sparse.jl")
