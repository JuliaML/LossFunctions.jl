
function center!(X::AbstractMatrix, μ::AbstractVector = vec(mean(X, 2)))
    k, n = size(X)
    @inbounds for j = 1:k
        for i = 1:n
            X[j, i] = X[j, i] - μ[j]
        end
    end
    μ
end

function rescale!(X::AbstractMatrix, σ::AbstractVector = vec(std(X, 2)))
    k, n = size(X)
    @inbounds for j = 1:k
        if σ[j] > 0
            for i = 1:n
                X[j, i] = X[j, i] / σ[j]
            end
        end
    end
    σ
end

function center_rescale!(X::AbstractMatrix, μ::AbstractVector = vec(mean(X, 2)), σ::AbstractVector = vec(std(X, 2)))
    k, n = size(X)
    @inbounds for j = 1:k
        for i = 1:n
            X[j, i] = X[j, i] - μ[j]
            if σ[j] > 0
                X[j, i] = X[j, i] / σ[j]
            end
        end
    end
    μ, σ
end

# ==========================================================================

immutable CenterRescale
    mean::Vector{Float64}
    std::Vector{Float64}

    function CenterRescale(m::Vector{Float64}, s::Vector{Float64})
        @_dimcheck length(m) == length(s)
        new(m, s)
    end
end

function CenterRescale{T<:Real}(X::AbstractMatrix{T})
    CenterRescale(vec(mean(X, 2)), vec(std(X, 2)))
end

function fit{T<:Real}(::Type{CenterRescale}, X::AbstractMatrix{T})
    CenterRescale(X)
end

function predict!{T<:Real}(cs::CenterRescale, X::AbstractMatrix{T})
    @_dimcheck length(cs.mean) == size(X, 1)
    center_rescale!(X, cs.mean, cs.std)
    X
end

function predict{T<:Real}(cs::CenterRescale, X::AbstractMatrix{T})
    Xnew = copy(X)
    predict!(cs, Xnew)
end
