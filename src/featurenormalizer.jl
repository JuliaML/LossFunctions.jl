immutable FeatureNormalizer
    offset::Vector{Float64}
    scale::Vector{Float64}

    function FeatureNormalizer(offset::Vector{Float64}, scale::Vector{Float64})
        @_dimcheck length(offset) == length(scale)
        new(offset, scale)
    end
end

function FeatureNormalizer{T<:Real}(X::AbstractMatrix{T})
    FeatureNormalizer(vec(mean(X, 2)), vec(std(X, 2)))
end

function fit{T<:Real}(::Type{FeatureNormalizer}, X::AbstractMatrix{T})
    FeatureNormalizer(X)
end

function predict!{T<:Real}(cs::FeatureNormalizer, X::AbstractMatrix{T})
    @_dimcheck length(cs.offset) == size(X, 1)
    DataUtils.normalize!(X, cs.offset, cs.scale)
    X
end

function predict{T<:Real}(cs::FeatureNormalizer, X::AbstractMatrix{T})
    Xnew = copy(X)
    predict!(cs, Xnew)
end
