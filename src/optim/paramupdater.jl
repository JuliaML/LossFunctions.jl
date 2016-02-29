
# -------------------------------------------------------------
# Repurposed from https://github.com/tbreloff/OnlineAI.jl
# -------------------------------------------------------------

abstract ParamUpdater
abstract ParamUpdaterState

# ----------------------------------------

doc"Stochastic Gradient Descent with Momentum"
type SGDUpdater <: ParamUpdater
  η::Float64 # learning rate
  μ::Float64 # momentum
  λ::Float64 # L2 penalty term
end
SGDUpdater(; η=0.1, μ=0.5, λ=1e-5) = SGDUpdater(η, μ, λ)

immutable SGDState{T <: AbstractArray} <: ParamUpdaterState
  lastChanges::T
end
SGDState(dims::Integer...) = SGDState(zeros(dims...))
gradient_state(model::SGDUpdater, dims::Integer...) = SGDState(dims...)

# update and return the change
function Δij(model::SGDUpdater, state::SGDState, gradient::Float64, val::Float64, i::Int, j::Int)
  state.lastChanges[i,j] = -model.η * (gradient + model.λ * val) + model.μ * state.lastChanges[i,j]
end

# ----------------------------------------

doc"Adaptive Gradient"
type AdagradUpdater <: ParamUpdater
  ε::Float64  # try 0.01?
  η::Float64 # base learning rate (numerator)
  λ::Float64 # L2 penalty term
end
AdagradUpdater(; ε=1e-8, η=1.0, λ=1e-6) = AdagradUpdater(ε, η, λ)

immutable AdagradState{T <: AbstractArray} <: ParamUpdaterState
  G::T
end
AdagradState(dims::Integer...) = AdagradState(zeros(dims...))

gradient_state(model::AdagradUpdater, dims::Integer...) = AdagradState(dims...)

function Δij(model::AdagradUpdater, state::AdagradState, gradient::Float64, val::Float64, i::Int, j::Int)
  state.G[i,j] += gradient^2
  η = model.η / sqrt(model.ε + state.G[i,j])
  -η * (gradient + model.λ * val)
end

# ----------------------------------------

doc"""
See: ADADELTA: An Adaptive Learning Rate Method (Zeiler 2012)

Relatively parameter-free... can probably avoid changing ε and ρ
"""
type AdadeltaUpdater <: ParamUpdater
  ε::Float64  # try 0.01?
  η::Float64
  ρ::Float64  # try 0.97?
  λ::Float64 # L2 penalty term
end
AdadeltaUpdater(; ε=1e-8, η=0.1, ρ=0.95, λ=1e-6) = AdadeltaUpdater(ε, η, ρ, λ)


immutable AdadeltaState{T <: AbstractArray} <: ParamUpdaterState
  dMean::T
  GMean::T
end
AdadeltaState(dims::Integer...) = AdadeltaState(zeros(dims...), zeros(dims...))
gradient_state(model::AdadeltaUpdater, dims::Integer...) = AdadeltaState(dims...)

function Δij(model::AdadeltaUpdater, state::AdadeltaState, gradient::Float64, val::Float64, i::Int, j::Int)
  ε, ρ = model.ε, model.ρ

  # average g²
  state.GMean[i,j] = ρ * state.GMean[i,j] + (1.0 - ρ) * gradient^2

  # compute learning rate from previous average dw² and current average g²
  η = model.η * sqrt(state.dMean[i,j] + ε) / sqrt(state.GMean[i,j] + ε)

  # compute change and update average dw²
  dij = -η * (gradient + model.λ * val)
  state.dMean[i,j] = ρ * state.dMean[i,j] + (1.0 - ρ) * dij^2
  dij
end

"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

Tracks an exponential moving average of the first and second moments of the gradient,
adjusting for zero-bias.  The defaults are those suggested in the paper.

TODO: AdaMax is similar, using the p-norm as p -> ∞
"""
type AdamUpdater <: ParamUpdater
  ε::Float64  # small number so we don't divide by 0
  η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
  ρ1::Float64 # decay for first moment (β₁ in the paper)
  ρ2::Float64 # decay for second moment (β₂ in the paper)
  λ::Float64  # L2 penalty term
end
AdamUpdater(; ε=1e-8, η=1e-3, ρ1=0.9, ρ2=0.999, λ=1e-6) = AdamUpdater(ε, η, ρ1, ρ2, λ)

type AdamState{T <: AbstractArray} <: ParamUpdaterState
  m::T # average first moment
  v::T # average second moment
  ρ1t::Float64  # β₁ᵗ from the paper... t-th power of β₁
  ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
end
AdamState(dims::Integer...) = AdamState(zeros(dims...), zeros(dims...), 1.0, 1.0)
gradient_state(model::AdamUpdater, dims::Integer...) = AdamState(dims...)

function Δij(model::AdamUpdater, state::AdamState, gradient::Float64, val::Float64, i::Int, j::Int)
  ρ1, ρ2 = model.ρ1, model.ρ2
  state.m[i,j] = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
  state.v[i,j] = ρ2 * state.v[i,j] + (1.0 - ρ2) * gradient^2
  if i == 1 && j == 1
    state.ρ1t *= model.ρ1
    state.ρ2t *= model.ρ2
  end
  ηt = model.η * (sqrt(1.0 - state.ρ2t) / (1.0 - state.ρ1t))
  -ηt * state.m[i,j] / (sqrt(state.v[i,j] + model.ε))
end


"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

AdaMax is similar to Adam, using the p-norm as p -> ∞
"""
type AdaMaxUpdater <: ParamUpdater
  # ε::Float64  # small number so we don't divide by 0
  η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
  ρ1::Float64 # decay for first moment (β₁ in the paper)
  ρ2::Float64 # decay for second moment (β₂ in the paper)
  λ::Float64  # L2 penalty term
end
AdaMaxUpdater(; η=1e-3, ρ1=0.9, ρ2=0.99, λ=1e-6) = AdaMaxUpdater(η, ρ1, ρ2, λ)

immutable AdaMaxState{T <: AbstractArray} <: ParamUpdaterState
  m::T # average first moment
  u::T # average second moment
  ρ1t::Vector{Float64}  # β₁ᵗ from the paper... t-th power of β₁
  # ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
end
AdaMaxState(dims::Integer...) = AdaMaxState(zeros(dims...), zeros(dims...), [1.0])
gradient_state(model::AdaMaxUpdater, dims::Integer...) = AdaMaxState(dims...)

function Δij(model::AdaMaxUpdater, state::AdaMaxState, gradient::Float64, val::Float64, i::Int, j::Int)
  # ρ1, ρ2 = model.ρ1, model.ρ2
  ρ1 = model.ρ1
  # state.m[i,j] = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
  mij = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
  state.m[i,j] = mij
  # state.u[i,j] = ρ2 * state.u[i,j] + (1.0 - ρ2) * gradient^2
  uij = max(model.ρ2 * state.u[i,j], abs(gradient))
  state.u[i,j] = uij
  if i == 1 && j == 1
    state.ρ1t[1] *= ρ1
  end
  # ηt = model.η / (1.0 - state.ρ1t[1])
  # -ηt * mij / (uij + 1e-10)
  -model.η * mij / ((uij + 1e-10) * (1.0 - state.ρ1t[1]))
end

# ----------------------------------------

