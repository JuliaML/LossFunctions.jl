# broadcasting behavior
Broadcast.broadcastable(loss::SupervisedLoss) = Ref(loss)

# fallback to unary evaluation
value(loss::DistanceLoss, target::Number, output::Number)  = value(loss, output - target)
deriv(loss::DistanceLoss, target::Number, output::Number)  = deriv(loss, output - target)
deriv2(loss::DistanceLoss, target::Number, output::Number) = deriv2(loss, output - target)

value(loss::MarginLoss, target::Number, output::Number)  = value(loss, target * output)
deriv(loss::MarginLoss, target::Number, output::Number)  = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, target::Number, output::Number) = deriv2(loss, target * output)

# result type when applying the loss to a single pair of objects
result_type(loss::SupervisedLoss, t::Type, o::Type) = typeof(value(loss, zero(t), zero(o)))

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

# helper macro (for devs)
macro dimcheck(condition)
    :(($(esc(condition))) || throw(DimensionMismatch("Dimensions of the parameters don't match: $($(string(condition)))")))
end

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
                ($($FUN)).(loss, target, output)
            end
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
                out = zero(result_type(loss, Q, T))
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i))
                end
                out
            end
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
            quote
                @nexprs $S (n)->@dimcheck(size(target, n) == size(output, n))
                nrm = 1 / length($bigger)
                out = zero(result_type(loss, Q, T)) * nrm
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i)) * nrm
                end
                out
            end
        end

        # ---------------------------
        # AGGREGATION: WEIGHTED SUM
        # ---------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                agg::AggMode.WeightedSum) where {Q,T,N}
            @dimcheck length(target) == length(output)
            nrm = agg.normalize ? inv(sum(agg.weights)) : inv(one(sum(agg.weights)))
            f(i) = nrm * agg.weights[i] * ($FUN)(loss, target[i], output[i])
            sum(f(i) for i in 1:length(output))
        end

        # ----------------------------
        # AGGREGATION: WEIGHTED MEAN
        # ----------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                agg::AggMode.WeightedMean) where {Q,T,N}
            @dimcheck length(target) == length(output)
            k = length(output)
            nrm = agg.normalize ? inv(k * sum(agg.weights)) : inv(k * one(sum(agg.weights)))
            f(i) = nrm * agg.weights[i] * ($FUN)(loss, target[i], output[i])
            sum(f(i) for i in 1:length(output))
        end
    end
end

# convenient functor interface
(loss::SupervisedLoss)(target::AbstractArray, output::AbstractArray) = value(loss, target, output)
