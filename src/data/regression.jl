module RegressionData

"""
`polynome(coef, x; noise = 0.0, f_rand = randn)`

Performs a (optionally noisy) polynomial basis expansion on the input vector  `x`.
The coefficients for the polynome are stored in the vector `coef`,
in which the first element denotes the coefficient for the largest element
"""
function polynome{T<:Real,R<:Real}(coef::AbstractVector{R}, x::AbstractVector{T}; noise::Real = 0.0, f_rand::Function = randn)
    n = length(x)
    m = length(coef)
    X = zeros(Float64, (m, n))
    x_vec = collect(x)
    @inbounds for i = 1:n
        for (d, k) in enumerate(coef)
            X[d, i] += x_vec[i]^(m - d)
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

function noisy_function{T<:Real}(f::Function, x::AbstractVector{T}; noise::Real = 0.01, f_rand::Function = randn)
    x_vec = collect(x)
    n = length(x_vec)
    y = f(x_vec) + noise * f_rand(n)
    x_vec, y
end

function example_sin{T<:Real}(x::AbstractVector{T} = 0:.05:2π; noise::Real = 0.3, f_rand::Function = randn)
    noisy_function(sin, x; noise = noise, f_rand = f_rand)
end

end
