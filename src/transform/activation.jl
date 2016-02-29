
# -------------------------------------------------------------
# Repurposed from https://github.com/tbreloff/OnlineAI.jl
# -------------------------------------------------------------

abstract Activation <: Transformation

value(activation::Activation, Σs::AVecF) = Float64[value(activation, Σ) for Σ in Σs]
deriv(activation::Activation, Σs::AVecF) = Float64[deriv(activation, Σ) for Σ in Σs]

function value!(activation::Activation, a::AVecF, Σs::AVecF)
  for i in 1:length(a)
    a[i] = value(activation, Σs[i])
  end
end
function deriv!(activation::Activation, a::AVecF, Σs::AVecF)
  for i in 1:length(a)
    a[i] = deriv(activation, Σs[i])
  end
end

# ---------------------------------------

doc"""
f(Σ) = Σ
f'(Σ) = 1
"""
immutable IdentityActivation <: Activation end
value(activation::IdentityActivation, Σ::Float64) = Σ
deriv(activation::IdentityActivation, Σ::Float64) = 1.0

doc"""
f(Σ) = 1 / (1 + exp(-Σ))
f'(Σ) = f(Σ) * (1 - f(Σ))
"""
immutable SigmoidActivation <: Activation end
value(activation::SigmoidActivation, Σ::Float64) = 1.0 / (1.0 + exp(-Σ))
deriv(activation::SigmoidActivation, Σ::Float64) = (s = value(activation, Σ); s * (1.0 - s))

doc"""
f(Σ) = tanh(Σ)
f'(Σ) = 1 - tanh(Σ)²
"""
immutable TanhActivation <: Activation end
value(activation::TanhActivation, Σ::Float64) = tanh(Σ)
deriv(activation::TanhActivation, Σ::Float64) = 1.0 - tanh(Σ)^2

doc"""
f(Σ) = Σ / (1 + |Σ|)
f'(Σ) = 1 / (1 + |Σ|)²
"""
immutable SoftsignActivation <: Activation end
value(activation::SoftsignActivation, Σ::Float64) = Σ / (1.0 + abs(Σ))
deriv(activation::SoftsignActivation, Σ::Float64) = 1.0 / (1.0 + abs(Σ))^2

# Rectified Linear Unit
doc"""
f(Σ) = max(0, Σ)
f'(Σ) = { 1,  Σ > 0
        { 0,  Σ ≤ 0
"""
immutable ReLUActivation <: Activation end
value(activation::ReLUActivation, Σ::Float64) = max(0.0, Σ)
deriv(activation::ReLUActivation, Σ::Float64) = float(Σ > 0.0)

# Leaky Rectified Linear Unit: modified derivative to fix "dying ReLU" problem
doc"""
f(Σ) = max(0, Σ)
f'(Σ) = { 1,  Σ > 0
        { ρ,  Σ ≤ 0
"""
immutable LReLUActivation <: Activation
  ρ::Float64
end
LReLUActivation() = LReLUActivation(0.01)
value(activation::LReLUActivation, Σ::Float64) = max(0.0, Σ)
deriv(activation::LReLUActivation, Σ::Float64) = Σ > 0.0 ? 1.0 : activation.ρ

doc"""
UNTESTED
f(Σᵢ) = exp(Σᵢ) / Z
  where Z := sum(exp(Σ))
f'(Σᵢ) = f(Σᵢ) * (1 - f(Σᵢ))

Note: we expect the target vector to be a multinomal indicator vector, where 
a 1 in the iᵗʰ position implies that the instance belongs to the iᵗʰ class
"""
immutable SoftmaxActivation <: Activation end

function value(activation::SoftmaxActivation, Σs::AVecF)
  evec = exp(Σs)
  evec / sum(evec)
end
function deriv(activation::SoftmaxActivation, Σs::AVecF)
  error("This should never actually be used, as we expect Softmax to be used only with CrossEntropyCostModel and so we don't multiply by the derivative")
end


