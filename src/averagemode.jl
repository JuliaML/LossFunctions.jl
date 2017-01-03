abstract AverageMode

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

    immutable None <: AverageMode end
    immutable Sum <: AverageMode end
    immutable Macro <: AverageMode end
    immutable Mean <: AverageMode end
    typealias Micro Mean

    immutable WeightedMean{T<:WeightVec} <: AverageMode
        weights::T
        normalize::Bool
    end
    WeightedMean(A::AbstractVector, normalize::Bool) = WeightedMean(weights(A), normalize)
    WeightedMean(A::AbstractVector; normalize::Bool = true) = WeightedMean(weights(A), normalize)

    immutable WeightedSum{T<:WeightVec} <: AverageMode
        weights::T
        normalize::Bool
    end
    WeightedSum(A::AbstractVector, normalize::Bool) = WeightedSum(weights(A), normalize)
    WeightedSum(A::AbstractVector; normalize::Bool = true) = WeightedSum(weights(A), normalize)

end

