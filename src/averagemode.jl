abstract type AverageMode end

module AvgMode

    using StatsBase
    import ..LossFunctions.AverageMode

    export
        None,
        Sum,
        Mean,
        Macro,
        Micro,
        WeightedMean,
        WeightedSum

    struct None <: AverageMode end
    struct Sum <: AverageMode end
    struct Macro <: AverageMode end
    struct Mean <: AverageMode end
    const Micro = Mean

    struct WeightedMean{T<:WeightVec} <: AverageMode
        weights::T
        normalize::Bool
    end
    WeightedMean(A::AbstractVector, normalize::Bool) = WeightedMean(weights(A), normalize)
    WeightedMean(A::AbstractVector; normalize::Bool = true) = WeightedMean(weights(A), normalize)

    struct WeightedSum{T<:WeightVec} <: AverageMode
        weights::T
        normalize::Bool
    end
    WeightedSum(A::AbstractVector, normalize::Bool) = WeightedSum(weights(A), normalize)
    WeightedSum(A::AbstractVector; normalize::Bool = false) = WeightedSum(weights(A), normalize)

end
