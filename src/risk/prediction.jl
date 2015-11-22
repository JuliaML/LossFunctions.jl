abstract Predictor{INTERCEPT}

intercept{INTERCEPT}(::Predictor{INTERCEPT}) = INTERCEPT

@inline call(p::Predictor, args...) = value(p, args...)
@inline transpose(p::Predictor) = grad_fun(p)

@inline function deriv_fun(p::Predictor)
    _deriv(args...) = deriv(p, args...)
    _deriv
end

@inline function grad_fun(p::Predictor)
    _grad(args...) = grad(p, args...)
    _grad
end

# ==========================================================================
# h(x,w) = wᵀx

immutable LinearPredictor{INTERCEPT} <: Predictor{INTERCEPT}
    bias::Float64
    function LinearPredictor(bias::Real)
        ((typeof(INTERCEPT) <: Bool) && (bias != 0.) == INTERCEPT) || throw(MethodError())
        new(Float64(bias))
    end
end

LinearPredictor(bias::Real) = LinearPredictor{bias!=0.}(bias)
LinearPredictor(;bias::Real = 1.) = LinearPredictor{bias!=0.}(bias)

@inline value(h::LinearPredictor, x::Number, w::Number) = x * w
@inline deriv(h::LinearPredictor, x::Number, w::Number) = x

# --------------------------------------------------------------------------
# no intercept

@inline value(h::LinearPredictor{false}, x::AbstractVector, w::AbstractVector) = dot(x, w)

@inline grad(h::LinearPredictor{false}, x::AbstractVector, w::AbstractVector) = x

@inline function value(h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVecOrMat)
    buffer = Array(Float64, size(w, 2), size(X, 2))
    value!(buffer, h, X, w)
    buffer
end

@inline function value!(buffer::AbstractMatrix, h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVecOrMat)
    k = size(w,2)
    n = size(X,2)
    @_dimcheck size(X, 1) == size(w, 1) && size(buffer) == (k, n)
    At_mul_B!(buffer, w, X)
    buffer
end

@inline grad(h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVector) = X

@inline function grad!(buffer::AbstractArray, h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVector)
    @_dimcheck size(buffer) == size(X)
    copy!(buffer, X)
    buffer
end

# --------------------------------------------------------------------------
# with intercept

@inline function value{T}(h::LinearPredictor{true}, x::AbstractVector, w::AbstractVector{T})
    k = length(w)-1
    @_dimcheck length(x) == k
    w⃗ = view(w, 1:k)
    @inbounds b = h.bias * w[k+1]
    dot(x, w⃗) + b
end

@inline function grad{T}(h::LinearPredictor{true}, x::AbstractVector, w::AbstractVector{T})
    k = length(w)-1
    @_dimcheck length(x) == k
    buffer = Array(T, length(w))
    for i = 1:k
        @inbounds buffer[i] = x[i]
    end
    buffer[end] = h.bias
    buffer
end

@inline function value{T}(h::LinearPredictor{true}, X::AbstractMatrix, w::AbstractVector{T})
    buffer = zeros(T, 1, size(X, 2))
    value!(buffer, h, X, w)
end

@inline function value!{T}(buffer::AbstractMatrix{T}, h::LinearPredictor{true}, X::AbstractMatrix, w::AbstractVector{T})
    k = length(w) - 1
    n = size(X, 2)
    @_dimcheck size(X, 1) == k && size(buffer) == (1, n)
    w⃗ = view(w, 1:k)
    @inbounds b = h.bias * w[k+1]
    At_mul_B!(buffer, w⃗, X)
    broadcast!(+, buffer, buffer, b)
    buffer
end

@inline grad{T}(h::LinearPredictor{true}, X::AbstractMatrix{T}, w::AbstractVector) = vcat(X, fill(convert(T,h.bias), (1, size(X,2))))

@inline function grad!{T}(buffer::AbstractMatrix{T}, h::LinearPredictor{true}, X::AbstractMatrix{T}, w::AbstractVector)
    k = size(X, 1)
    k1 = k + 1
    n = size(X, 2)
    @_dimcheck size(buffer) == (k1, n)
    bias = convert(T, h.bias)
    @inbounds for i = 1:n
        for j = 1:k
            buffer[j, i] = X[j, i]
        end
        buffer[k1, i] = bias
    end
    buffer
end

# ==========================================================================
# h(x,w) = 1 / (1 + exp(-f(x,w))

@inline sigmoid(x) = one(x) ./ (one(x) + exp(-x))

immutable SigmoidPredictor{INTERCEPT} <: Predictor{INTERCEPT}
    f::LinearPredictor{INTERCEPT}
end

SigmoidPredictor{T}(f::LinearPredictor{T} = LinearPredictor()) =
    SigmoidPredictor{T}(f)

@inline value(h::SigmoidPredictor, x, w) = sigmoid(value(h.f, x, w))

@inline function deriv(h::SigmoidPredictor, x, w)
    z = value(h, x, w)
    (z .* (1 - z)) .* deriv(h.f, x, w)
end
