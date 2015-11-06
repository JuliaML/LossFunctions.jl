abstract Predictor

@inline call(p::Predictor, args...) = value(p, args...)
@inline transpose(p::Predictor) = grad_fun(p)

function deriv_fun(p::Predictor)
  _deriv(args...) = deriv(p, args...)
  _deriv
end

function grad_fun(p::Predictor)
  _grad(args...) = grad(p, args...)
  _grad
end

# ==========================================================================
# h(x,w) = wᵀx

immutable LinearPredictor{INTERCEPT} <: Predictor
  bias::Float64
  function LinearPredictor(bias::Real)
    ((typeof(INTERCEPT) <: Bool) && (bias != 0.) == INTERCEPT) || throw(MethodError())
    new(Float64(bias))
  end
end

LinearPredictor(bias::Real) = LinearPredictor{bias!=0.}(bias)
LinearPredictor(;bias::Real = 1.) = LinearPredictor{bias!=0.}(bias)

value(h::LinearPredictor{false}, x::Number, w::Number) = x * w
deriv(h::LinearPredictor{false}, x::Number, w::Number) = x
value(h::LinearPredictor{false}, x::AbstractVector, w::AbstractVector) = dot(x, w)
grad(h::LinearPredictor{false}, x::AbstractVector, w::AbstractVector) = x

function value(h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVecOrMat)
  k = size(w,2)
  n = size(X,2)
  @_dimcheck size(X, 1) == size(w, 1)
  buffer = Array(Float64, size(w, 2), size(X, 2))
  At_mul_B!(buffer, w, X)
  buffer
end

function value!(buffer::AbstractMatrix, h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVecOrMat)
  k = size(w,2)
  n = size(X,2)
  @_dimcheck size(X, 1) == size(w, 1) && size(buffer) == (k, n)
  At_mul_B!(buffer, w, X)
  buffer
end

grad(h::LinearPredictor{false}, X::AbstractMatrix, w::AbstractVector) = X

# ==========================================================================

value(h::LinearPredictor{true}, x::Number, w::AbstractVector) = x * w[1] + h.bias * w[2] 
grad(h::LinearPredictor{true}, x::Number, w::AbstractVector) = [x, h.bias]

@inline function value(h::LinearPredictor{true}, x::AbstractVector, w::AbstractVector)
  k = length(w)-1
  @_dimcheck length(x) == k
  w⃗ = slice(w, 1:k)
  b = convert(T, h.bias) * w[k+1]
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
end

@inline function value{T}(h::LinearPredictor{true}, X::AbstractMatrix, w::AbstractVector{T})
  buffer = zeros(T, 1, size(X, 2))
  value!(buffer, h, X, w)
end

@inline function value!{T}(buffer::AbstractMatrix, h::LinearPredictor{true}, X::AbstractMatrix, w::AbstractVector{T})
  k = length(w) - 1
  n = size(X, 2)
  @_dimcheck size(X, 1) == k && size(buffer) == (1, n)
  w⃗ = slice(w, 1:k)
  b = convert(T, h.bias) * w[k+1]
  At_mul_B!(buffer, w⃗, X)
  broadcast!(+, buffer, buffer, b)
  buffer
end

grad(h::LinearPredictor{true}, X::AbstractMatrix, w::AbstractVector) = vcat(X, ones(1, size(X,2)))

grad!(h::LinearPredictor{true}, X::AbstractMatrix, w::AbstractVector) = vcat(X, ones(1, size(X,2)))

# ==========================================================================
# h(x,w) = 1 / (1 + exp(-f(x,w))

sigmoid(x) = one(x) ./ (one(x) + exp(-x))

immutable SigmoidPredictor{T<:Predictor} <: Predictor
  f::T
end

SigmoidPredictor(f = LinearPredictor()) = SigmoidPredictor{typeof(f)}(f)
value(h::SigmoidPredictor, x, w) = sigmoid(value(h.f, x, w))
function deriv(h::SigmoidPredictor, x, w)
  z = value(h, x, w)
  (z .* (1 - z)) .* deriv(h.f, x, w)
end
