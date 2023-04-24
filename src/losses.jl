# broadcasting behavior
Broadcast.broadcastable(loss::SupervisedLoss) = Ref(loss)

# fallback to unary evaluation
value(loss::DistanceLoss, output::Number, target::Number)  = value(loss, output - target)
deriv(loss::DistanceLoss, output::Number, target::Number)  = deriv(loss, output - target)
deriv2(loss::DistanceLoss, output::Number, target::Number) = deriv2(loss, output - target)

value(loss::MarginLoss, output::Number, target::Number)  = value(loss, target * output)
deriv(loss::MarginLoss, output::Number, target::Number)  = target * deriv(loss, target * output)
deriv2(loss::MarginLoss, output::Number, target::Number) = deriv2(loss, target * output)

# ------------------
# AVAILABLE LOSSES
# ------------------
include("losses/distance.jl")
include("losses/margin.jl")
include("losses/other.jl")

# meta-losses
include("losses/scaled.jl")
include("losses/weighted.jl")

# ----------------------
# AGGREGATION BEHAVIOR
# ----------------------
for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # ------------------
        # AGGREGATION: SUM
        # ------------------
        function ($FUN)(
                loss::SupervisedLoss,
                outputs::AbstractVector,
                targets::AbstractVector,
                ::AggMode.Sum)
            nobs = length(outputs)
            f(i) = ($FUN)(loss, outputs[i], targets[i])
            sum(f, 1:nobs)
        end

        # -------------------
        # AGGREGATION: MEAN
        # -------------------
        function ($FUN)(
                loss::SupervisedLoss,
                outputs::AbstractVector,
                targets::AbstractVector,
                ::AggMode.Mean)
            nobs = length(outputs)
            f(i) = ($FUN)(loss, outputs[i], targets[i])
            sum(f, 1:nobs) / nobs
        end

        # ---------------------------
        # AGGREGATION: WEIGHTED SUM
        # ---------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                outputs::AbstractVector,
                targets::AbstractVector,
                agg::AggMode.WeightedSum)
            nobs  = length(outputs)
            wsum  = sum(agg.weights)
            denom = agg.normalize ? wsum : one(wsum)
            f(i)  = agg.weights[i] * ($FUN)(loss, outputs[i], targets[i])
            sum(f, 1:nobs) / denom
        end

        # ----------------------------
        # AGGREGATION: WEIGHTED MEAN
        # ----------------------------
        function ($FUN)(
                loss::SupervisedLoss,
                outputs::AbstractVector,
                targets::AbstractVector,
                agg::AggMode.WeightedMean)
            nobs  = length(outputs)
            wsum  = sum(agg.weights)
            denom = agg.normalize ? nobs * wsum : nobs * one(wsum)
            f(i)  = agg.weights[i] * ($FUN)(loss, outputs[i], targets[i])
            sum(f, 1:nobs) / denom
        end
    end
end

# convenient functor interface
(loss::SupervisedLoss)(outputs::AbstractVector, targets::AbstractVector) = value.(loss, outputs, targets)
