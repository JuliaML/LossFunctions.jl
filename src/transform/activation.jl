
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
value(activation::IdentityActivation, input::Real) = input
deriv(activation::IdentityActivation, input::Real) = 1.0

# ----------------------------------------------------------------------------

doc"""
f(x) = 1 / (1 + exp(-x))
f'(x) = f(x) * (1 - f(x))
"""
immutable SigmoidActivation <: Activation end
value(activation::SigmoidActivation, input::Real) = 1.0 / (1.0 + exp(-input))
deriv(activation::SigmoidActivation, input::Real) = (s = value(activation, input); s * (1.0 - s))

# ----------------------------------------------------------------------------

doc"""
f(x) = tanh(x)
f'(x) = 1 - tanh(x)²
"""
immutable TanhActivation <: Activation end
value(activation::TanhActivation, input::Real) = tanh(input)
deriv(activation::TanhActivation, input::Real) = 1.0 - tanh(input)^2

# ----------------------------------------------------------------------------

doc"""
f(x) = x / (1 + |x|)
f'(x) = 1 / (1 + |x|)²
"""
immutable SoftsignActivation <: Activation end
value(activation::SoftsignActivation, input::Real) = input / (1.0 + abs(input))
deriv(activation::SoftsignActivation, input::Real) = 1.0 / (1.0 + abs(input))^2

# ----------------------------------------------------------------------------

# Rectified Linear Unit
doc"""
f(x) = max(0, x)
f'(x) = { 1,  x > 0
        { 0,  x ≤ 0
"""
immutable ReLUActivation <: Activation end
value(activation::ReLUActivation, input::Real) = max(0.0, input)
deriv(activation::ReLUActivation, input::Real) = float(input >= 0.0)

# ----------------------------------------------------------------------------

# Leaky Rectified Linear Unit: modified derivative to fix "dying ReLU" problem
doc"""
f(x) = max(0, x)
f'(x) = { 1,  x > 0
        { ρ,  x ≤ 0
"""
immutable LReLUActivation <: Activation
    ρ::Float64
end
LReLUActivation() = LReLUActivation(0.01)
value(activation::LReLUActivation, input::Real) = deriv(activation, input) * input 
deriv{T<:Real}(activation::LReLUActivation, input::T) = input >= zero(T) ? one(T) : activation.ρ

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
    evec / sum(evec)
end

function deriv{T<:Real}(activation::SoftmaxActivation, input::AbstractVector{T})
    buffer = Array(T, length(input))
    deriv!(buffer, activation, input)
end

function value!(buffer::AbstractVector, activation::SoftmaxActivation, input::AbstractVector)
    buffer[:] = exp(input)
    s = sum(buffer)
    for i=1:length(input)
        buffer[i] /= s
    end
    buffer
end

function deriv!{T<:Real}(buffer::AbstractVector{T}, activation::SoftmaxActivation, input::AbstractVector{T})
    throw(ArgumentError("This should never actually be used, as we expect Softmax to be used only with CrossEntropyCostModel and so we don't multiply by the derivative"))

    # though in case we change our minds on this logic, here's the derivative:
    for i=1:length(input)
        buffer[i] = input[i] * (one(T) - input[i])
    end
    buffer
end



