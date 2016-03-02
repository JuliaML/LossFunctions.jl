
# -------------------------------------------------------------
# Repurposed from https://github.com/tbreloff/OnlineAI.jl
# -------------------------------------------------------------

abstract Mapping <: Transformation

value(mapping::Mapping, input::AbstractVector) = Float64[value(mapping, Σ) for Σ in input]
deriv(mapping::Mapping, input::AbstractVector) = Float64[deriv(mapping, Σ) for Σ in input]

function value!(buffer::AbstractVector, mapping::Mapping, input::AbstractVector)
    for i in 1:length(buffer)
        buffer[i] = value(mapping, input[i])
    end
end
function deriv!(buffer::AbstractVector, mapping::Mapping, input::AbstractVector)
    for i in 1:length(buffer)
        buffer[i] = deriv(mapping, input[i])
    end
end

# ----------------------------------------------------------------------------

doc"""
f(x) = x
f'(x) = 1
"""
immutable IdentityMapping <: Mapping end
value(mapping::IdentityMapping, input::Number) = input
deriv{T<:Number}(mapping::IdentityMapping, input::T) = one(T)

# ----------------------------------------------------------------------------

doc"""
f(x) = 1 / (1 + exp(-x))
f'(x) = f(x) * (1 - f(x))
"""
immutable SigmoidMapping <: Mapping end
value{T<:Number}(mapping::SigmoidMapping, input::T) = one(T) / (one(T) + exp(-input))
deriv{T<:Number}(mapping::SigmoidMapping, input::T) = (s = value(mapping, input); s * (one(T) - s))

# ----------------------------------------------------------------------------

doc"""
f(x) = tanh(x)
f'(x) = 1 - tanh(x)²
"""
immutable TanhMapping <: Mapping end
value(mapping::TanhMapping, input::Number) = tanh(input)
deriv{T<:Number}(mapping::TanhMapping, input::T) = one(T) - tanh(input)^2

# ----------------------------------------------------------------------------

doc"""
f(x) = x / (1 + |x|)
f'(x) = 1 / (1 + |x|)²
"""
immutable SoftsignMapping <: Mapping end
value{T<:Number}(mapping::SoftsignMapping, input::T) = input / (one(T) + abs(input))
deriv{T<:Number}(mapping::SoftsignMapping, input::T) = one(T) / (one(T) + abs(input))^2

# ----------------------------------------------------------------------------

# Rectified Linear Unit
doc"""
f(x) = max(0, x)
f'(x) = { 1,  x > 0
        { 0,  x ≤ 0
"""
immutable ReLUMapping <: Mapping end
value{T<:Number}(mapping::ReLUMapping, input::T) = max(zero(T), input)
deriv{T<:Number}(mapping::ReLUMapping, input::T) = input >= zero(T) ? one(T) : zero(T)

# ----------------------------------------------------------------------------

# Leaky Rectified Linear Unit: modified derivative to fix "dying ReLU" problem
doc"""
f(x) = max(0, x)
f'(x) = { 1,  x > 0
        { ρ,  x ≤ 0
"""
immutable LReLUMapping{T<:Number} <: Mapping
    ρ::T
end
LReLUMapping() = LReLUMapping(0.01)

value(mapping::LReLUMapping, input::Number) = deriv(mapping, input) * input 
deriv{T<:Number}(mapping::LReLUMapping{T}, input::T) = input >= zero(T) ? one(T) : mapping.ρ

function deriv{T<:Number, R<:Number}(mapping::LReLUMapping{T}, input::R)
    N = promote_type(T, R)
    input >= zero(R) ? one(N) : N(mapping.ρ)
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
immutable SoftmaxMapping <: Mapping end

function value(mapping::SoftmaxMapping, input::AbstractVector)
    evec = exp(input)
    s = sum(evec)
    for i in eachindex(evec)
        evec[i] /= s
    end
    evec
end

function deriv{T<:Number}(mapping::SoftmaxMapping, input::AbstractVector{T})
    buffer = Array(T, length(input))
    deriv!(buffer, mapping, input)
end

function value!{T<:Number}(buffer::AbstractVector{T}, mapping::SoftmaxMapping, input::AbstractVector{T})
    broadcast!(exp, buffer, input)
    s = sum(buffer)
    for i in eachindex(input)
        buffer[i] /= s
    end
    buffer
end

function deriv!{T<:Number}(buffer::AbstractVector{T}, mapping::SoftmaxMapping, input::AbstractVector{T})
    throw(ArgumentError("This should never actually be used, as we expect Softmax to be used only with CrossEntropyCostModel and so we don't multiply by the derivative"))

    # though in case we change our minds on this logic, here's the derivative:
    for i in eachindex(input)
        buffer[i] = input[i] * (one(T) - input[i])
    end
    buffer
end



