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
            f(i) = ($FUN)(loss, target[i], output[i])
            sum(f(i) for i in 1:length(output))
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
            nrm = inv(length(output))
            f(i) = nrm * ($FUN)(loss, target[i], output[i])
            sum(f(i) for i in 1:length(output))
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
            s = sum(agg.weights)
            nrm = agg.normalize ? inv(s) : inv(one(s))
            f(i) = nrm * agg.weights[i] * ($FUN)(loss, target[i], output[i])
            sum(f(i) for i in 1:length(output))
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
            k = length(output)
            s = sum(agg.weights)
            nrm = agg.normalize ? inv(k * s) : inv(k * one(s))
            f(i) = nrm * agg.weights[i] * ($FUN)(loss, target[i], output[i])
            sum(f(i) for i in 1:length(output))
        end
    end
end

# convenient functor interface
(loss::SupervisedLoss)(target::AbstractArray, output::AbstractArray) = value(loss, target, output)
