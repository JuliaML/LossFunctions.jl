
# -------------------------------------------------------------
# Repurposed from https://github.com/tbreloff/OnlineAI.jl
# -------------------------------------------------------------

abstract Activation <: Transformation

value(activation::Activation, input::AbstractVector) = Float64[value(activation, Σ) for Σ in input]
deriv(activation::Activation, input::AbstractVector) = Float64[deriv(activation, Σ) for Σ in input]

function value!(buffer::AbstractVector, activation::Activation, input::AbstractVector)
    for i in 1:length(buffer)
        buffer[i] = value(activation, input[i])
    end
end
function deriv!(buffer::AbstractVector, activation::Activation, input::AbstractVector)
    for i in 1:length(buffer)
        buffer[i] = deriv(activation, input[i])
    end
end

# ----------------------------------------------------------------------------

doc"""
f(x) = x
f'(x) = 1
"""
immutable IdentityActivation <: Activation end
value(activation::IdentityActivation, input::Number) = input
deriv{T<:Number}(activation::IdentityActivation, input::T) = one(T)

# ----------------------------------------------------------------------------

doc"""
f(x) = 1 / (1 + exp(-x))
f'(x) = f(x) * (1 - f(x))
"""
immutable SigmoidActivation <: Activation end
value{T<:Number}(activation::SigmoidActivation, input::T) = one(T) / (one(T) + exp(-input))
deriv{T<:Number}(activation::SigmoidActivation, input::T) = (s = value(activation, input); s * (one(T) - s))

# ----------------------------------------------------------------------------

doc"""
f(x) = tanh(x)
f'(x) = 1 - tanh(x)²
"""
immutable TanhActivation <: Activation end
value(activation::TanhActivation, input::Number) = tanh(input)
deriv{T<:Number}(activation::TanhActivation, input::T) = one(T) - tanh(input)^2

# ----------------------------------------------------------------------------

doc"""
f(x) = x / (1 + |x|)
f'(x) = 1 / (1 + |x|)²
"""
immutable SoftsignActivation <: Activation end
value{T<:Number}(activation::SoftsignActivation, input::T) = input / (one(T) + abs(input))
deriv{T<:Number}(activation::SoftsignActivation, input::T) = one(T) / (one(T) + abs(input))^2

# ----------------------------------------------------------------------------

# Rectified Linear Unit
doc"""
f(x) = max(0, x)
f'(x) = { 1,  x > 0
        { 0,  x ≤ 0
"""
immutable ReLUActivation <: Activation end
value{T<:Number}(activation::ReLUActivation, input::T) = max(zero(T), input)
deriv{T<:Number}(activation::ReLUActivation, input::T) = input >= zero(T) ? one(T) : zero(T)

# ----------------------------------------------------------------------------

# Leaky Rectified Linear Unit: modified derivative to fix "dying ReLU" problem
doc"""
f(x) = max(0, x)
f'(x) = { 1,  x > 0
        { ρ,  x ≤ 0
"""
immutable LReLUActivation{T<:Number} <: Activation
    ρ::T
end
LReLUActivation() = LReLUActivation(0.01)

value(activation::LReLUActivation, input::Number) = deriv(activation, input) * input 
deriv{T<:Number}(activation::LReLUActivation{T}, input::T) = input >= zero(T) ? one(T) : activation.ρ

function deriv{T<:Number, R<:Number}(activation::LReLUActivation{T}, input::R)
    N = promote_type(T, R)
    input >= zero(R) ? one(N) : N(activation.ρ)
end

# ----------------------------------------------------------------------------

doc"""
UNTESTED
f(xᵢ) = exp(xᵢ) / Z
  where Z := sum(exp(x))
f'(xᵢ) = f(xᵢ) * (1 - f(xᵢ))

Note: we expect the target vector to be a multinomal indicator vector, where 
a 1 in the iᵗʰ position implies that the instance belongs to the iᵗʰ class
"""
immutable SoftmaxActivation <: Activation end

function value(activation::SoftmaxActivation, input::AbstractVector)
    evec = exp(input)
    s = sum(evec)
    for i in eachindex(evec)
        evec[i] /= s
    end
    evec
end

function deriv{T<:Number}(activation::SoftmaxActivation, input::AbstractVector{T})
    buffer = Array(T, length(input))
    deriv!(buffer, activation, input)
end

function value!{T<:Number}(buffer::AbstractVector{T}, activation::SoftmaxActivation, input::AbstractVector{T})
    broadcast!(exp, buffer, input)
    s = sum(buffer)
    for i in eachindex(input)
        buffer[i] /= s
    end
    buffer
end

function deriv!{T<:Number}(buffer::AbstractVector{T}, activation::SoftmaxActivation, input::AbstractVector{T})
    throw(ArgumentError("This should never actually be used, as we expect Softmax to be used only with CrossEntropyCostModel and so we don't multiply by the derivative"))

    # though in case we change our minds on this logic, here's the derivative:
    for i in eachindex(input)
        buffer[i] = input[i] * (one(T) - input[i])
    end
    buffer
end



