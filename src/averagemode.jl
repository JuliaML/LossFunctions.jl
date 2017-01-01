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
        Weighted

    immutable None <: AverageMode end
    immutable Sum <: AverageMode end
    immutable Macro <: AverageMode end
    immutable Micro <: AverageMode end
    immutable Weighted{T<:WeightVec} <: AverageMode
        weights::T
    end
    Weighted(A::AbstractVector) = Weighted(weights(A))

end

