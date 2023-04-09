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
                targets::AbstractVector,
                outputs::AbstractVector)
            ($FUN)(loss, targets, outputs, AggMode.None())
        end

        # -------------------
        # AGGREGATION: NONE
        # -------------------
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractVector,
                output::AbstractVector,
                ::AggMode.None)
            quote
                $(Expr(:meta, :inline))
                ($($FUN)).(loss, target, output)
            end
        end

        # ------------------
        # AGGREGATION: SUM
        # ------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractVector,
                output::AbstractVector,
                ::AggMode.Sum)
            @dimcheck length(target) == length(output)
            nobs = length(output)
            f(i) = ($FUN)(loss, target[i], output[i])
            sum(f, 1:nobs)
        end

        # -------------------
        # AGGREGATION: MEAN
        # -------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractVector,
                output::AbstractVector,
                ::AggMode.Mean)
            @dimcheck length(target) == length(output)
            nobs = length(output)
            f(i) = ($FUN)(loss, target[i], output[i])
            sum(f, 1:nobs) / nobs
        end

        # ---------------------------
        # AGGREGATION: WEIGHTED SUM
        # ---------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractVector,
                output::AbstractVector,
                agg::AggMode.WeightedSum)
            @dimcheck length(target) == length(output)
            @dimcheck length(output) == length(agg.weights)
            nobs  = length(output)
            wsum  = sum(agg.weights)
            denom = agg.normalize ? wsum : one(wsum)
            f(i)  = agg.weights[i] * ($FUN)(loss, target[i], output[i])
            sum(f, 1:nobs) / denom
        end

        # ----------------------------
        # AGGREGATION: WEIGHTED MEAN
        # ----------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractVector,
                output::AbstractVector,
                agg::AggMode.WeightedMean)
            @dimcheck length(target) == length(output)
            @dimcheck length(output) == length(agg.weights)
            nobs  = length(output)
            wsum  = sum(agg.weights)
            denom = agg.normalize ? nobs * wsum : nobs * one(wsum)
            f(i)  = agg.weights[i] * ($FUN)(loss, target[i], output[i])
            sum(f, 1:nobs) / denom
        end
    end
end

# convenient functor interface
(loss::SupervisedLoss)(target::AbstractArray, output::AbstractArray) = value(loss, target, output)
