module RegressionData

function polynome{T<:Real,R<:Real}(x::AbstractVector{T}, coef::AbstractVector{R}; noise::Real = 0.01, f_rand::Function = randn)
    n = length(x)
    m = length(coef)
    X = zeros(Float64, (m, n))
    @inbounds for i = 1:n
        for (d, k) in enumerate(coef)
            X[d, i] += x[i]^(m - d)
        end
    end
    y = noise .* f_rand(n)
    @inbounds for i = 1:n
        for (d, k) in enumerate(coef)
            y[i] += k * X[d, i]
        end
    end
    X[1:(m-1),:], y
end

end
