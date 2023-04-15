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
include("supervised/distance.jl")
include("supervised/margin.jl")
include("supervised/other.jl")

# meta-losses
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
                outputs::AbstractVector,
                targets::AbstractVector)
            ($FUN)(loss, outputs, targets, AggMode.None())
        end

        # -------------------
        # AGGREGATION: NONE
        # -------------------
        @generated function ($FUN)(
                loss::SupervisedLoss,
                outputs::AbstractVector,
                targets::AbstractVector,
                ::AggMode.None)
            quote
                $(Expr(:meta, :inline))
                ($($FUN)).(loss, outputs, targets)
            end
        end

        # ------------------
        # AGGREGATION: SUM
        # ------------------
        function ($FUN)(
                loss::SupervisedLoss,
                outputs::AbstractVector,
                targets::AbstractVector,
                ::AggMode.Sum)
            @dimcheck length(outputs) == length(targets)
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
            @dimcheck length(outputs) == length(targets)
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
            @dimcheck length(outputs) == length(targets)
            @dimcheck length(outputs) == length(agg.weights)
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
            @dimcheck length(outputs) == length(targets)
            @dimcheck length(outputs) == length(agg.weights)
            nobs  = length(outputs)
            wsum  = sum(agg.weights)
            denom = agg.normalize ? nobs * wsum : nobs * one(wsum)
            f(i)  = agg.weights[i] * ($FUN)(loss, outputs[i], targets[i])
            sum(f, 1:nobs) / denom
        end
    end
end

# convenient functor interface
(loss::SupervisedLoss)(outputs::AbstractVector, targets::AbstractVector) = value(loss, outputs, targets)
