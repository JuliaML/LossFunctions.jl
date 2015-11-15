
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
        if σ[j] > 1
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
            if σ[j] > 1
                X[j, i] = X[j, i] / σ[j]
            end
        end
    end
    μ, σ
end
