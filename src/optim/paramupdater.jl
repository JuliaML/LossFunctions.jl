
# -------------------------------------------------------------
# Repurposed from https://github.com/tbreloff/OnlineAI.jl
# -------------------------------------------------------------

export
  ParameterUpdater,
  SGDUpdater,
  AdagradUpdater,
  AdadeltaUpdater,
  AdamUpdater,
  AdaMaxUpdater,

  ParameterUpdaterState,
  SGDState,
  AdagradState,
  AdadeltaState,
  AdamState,
  AdaMaxState,

  param_state

abstract ParameterUpdater
abstract ParameterUpdaterState

# -------------------------------------------------------------

"Stochastic Gradient Descent with Momentum"
type SGDUpdater{PLOSS <: ParameterLoss} <: ParameterUpdater
  η::Float64 # learning rate
  μ::Float64 # momentum
  ploss::PLOSS
end
SGDUpdater(; η=0.1, μ=0.5, ploss = NoParameterLoss()) = SGDUpdater(η, μ, ploss)

immutable SGDState{T <: AbstractVecOrMat} <: ParameterUpdaterState
  lastChanges::T
end
SGDState(dims::Integer...) = SGDState(zeros(dims...))
param_state(updater::SGDUpdater, dims::Integer...) = SGDState(dims...)

"Calculate the amount `Δwᵢⱼ` to adjust the parameter."
function param_change!(updater::SGDUpdater, state::SGDState, gradient::Real, param::Real, i::Integer, j::Integer)
  gradient += deriv(updater.ploss, param)
  state.lastChanges[i,j] = -updater.η * gradient + updater.μ * state.lastChanges[i,j]
end

# -------------------------------------------------------------

"Adaptive Gradient"
type AdagradUpdater{PLOSS <: ParameterLoss} <: ParameterUpdater
  ε::Float64  # try 0.01?
  η::Float64 # base learning rate (numerator)
  ploss::PLOSS
end
AdagradUpdater(; ε=1e-8, η=1.0, ploss=NoParameterLoss()) = AdagradUpdater(ε, η, ploss)

immutable AdagradState{T <: AbstractVecOrMat} <: ParameterUpdaterState
  G::T
end
AdagradState(dims::Integer...) = AdagradState(zeros(dims...))
param_state(updater::AdagradUpdater, dims::Integer...) = AdagradState(dims...)

function param_change!(updater::AdagradUpdater, state::AdagradState, gradient::Real, param::Real, i::Integer, j::Integer)
  gradient += deriv(updater.ploss, param)
  state.G[i,j] += gradient^2
  η = updater.η / sqrt(updater.ε + state.G[i,j])
  -η * gradient
end

# -------------------------------------------------------------

"""
See: ADADELTA: An Adaptive Learning Rate Method (Zeiler 2012)

Relatively parameter-free... can probably avoid changing ε and ρ
"""
type AdadeltaUpdater{PLOSS <: ParameterLoss} <: ParameterUpdater
  ε::Float64  # try 0.01?
  η::Float64
  ρ::Float64  # try 0.97?
  ploss::PLOSS
end
AdadeltaUpdater(; ε=1e-8, η=0.1, ρ=0.95, ploss=NoParameterLoss()) = AdadeltaUpdater(ε, η, ρ, ploss)


immutable AdadeltaState{T <: AbstractVecOrMat} <: ParameterUpdaterState
  dMean::T
  GMean::T
end
AdadeltaState(dims::Integer...) = AdadeltaState(zeros(dims...), zeros(dims...))
param_state(updater::AdadeltaUpdater, dims::Integer...) = AdadeltaState(dims...)

function param_change!(updater::AdadeltaUpdater, state::AdadeltaState, gradient::Real, param::Real, i::Integer, j::Integer)
  gradient += deriv(updater.ploss, param)
  ε, ρ = updater.ε, updater.ρ

  # average g²
  state.GMean[i,j] = ρ * state.GMean[i,j] + (1.0 - ρ) * gradient^2

  # compute learning rate from previous average dw² and current average g²
  η = updater.η * sqrt(state.dMean[i,j] + ε) / sqrt(state.GMean[i,j] + ε)

  # compute change and update average dw²
  dij = -η * gradient
  state.dMean[i,j] = ρ * state.dMean[i,j] + (1.0 - ρ) * dij^2
  dij
end

# -------------------------------------------------------------


"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

Tracks an exponential moving average of the first and second moments of the gradient,
adjusting for zero-bias.  The defaults are those suggested in the paper.

TODO: AdaMax is similar, using the p-norm as p -> ∞
"""
type AdamUpdater{PLOSS <: ParameterLoss} <: ParameterUpdater
  ε::Float64  # small number so we don't divide by 0
  η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
  ρ1::Float64 # decay for first moment (β₁ in the paper)
  ρ2::Float64 # decay for second moment (β₂ in the paper)
  ploss::PLOSS
end
AdamUpdater(; ε=1e-8, η=1e-3, ρ1=0.9, ρ2=0.999, ploss=NoParameterLoss()) = AdamUpdater(ε, η, ρ1, ρ2, ploss)

type AdamState{T <: AbstractVecOrMat} <: ParameterUpdaterState
  m::T # average first moment
  v::T # average second moment
  ρ1t::Float64  # β₁ᵗ from the paper... t-th power of β₁
  ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
end
AdamState(dims::Integer...) = AdamState(zeros(dims...), zeros(dims...), 1.0, 1.0)
param_state(updater::AdamUpdater, dims::Integer...) = AdamState(dims...)

function param_change!(updater::AdamUpdater, state::AdamState, gradient::Real, param::Real, i::Integer, j::Integer)
  gradient += deriv(updater.ploss, param)
  ρ1, ρ2 = updater.ρ1, updater.ρ2
  state.m[i,j] = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
  state.v[i,j] = ρ2 * state.v[i,j] + (1.0 - ρ2) * gradient^2
  if i == 1 && j == 1
    state.ρ1t *= updater.ρ1
    state.ρ2t *= updater.ρ2
  end
  ηt = updater.η * (sqrt(1.0 - state.ρ2t) / (1.0 - state.ρ1t))
  -ηt * state.m[i,j] / (sqrt(state.v[i,j] + updater.ε))
end

# -------------------------------------------------------------

"""
see: ADAM: A method for Stochastic Optimization (Kingma and Ba 2015)

AdaMax is similar to Adam, using the p-norm as p -> ∞
"""
type AdaMaxUpdater{PLOSS <: ParameterLoss} <: ParameterUpdater
  # ε::Float64  # small number so we don't divide by 0
  η::Float64  # learning rate... (this is α in the paper) maybe use around 1e-3?
  ρ1::Float64 # decay for first moment (β₁ in the paper)
  ρ2::Float64 # decay for second moment (β₂ in the paper)
  ploss::PLOSS
end
AdaMaxUpdater(; η=1e-3, ρ1=0.9, ρ2=0.99, ploss=NoParameterLoss()) = AdaMaxUpdater(η, ρ1, ρ2, ploss)

immutable AdaMaxState{T <: AbstractVecOrMat} <: ParameterUpdaterState
  m::T # average first moment
  u::T # average second moment
  ρ1t::Vector{Float64}  # β₁ᵗ from the paper... t-th power of β₁
  # ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
end
AdaMaxState(dims::Integer...) = AdaMaxState(zeros(dims...), zeros(dims...), [1.0])
param_state(updater::AdaMaxUpdater, dims::Integer...) = AdaMaxState(dims...)

function param_change!(updater::AdaMaxUpdater, state::AdaMaxState, gradient::Real, param::Real, i::Integer, j::Integer)
  gradient += deriv(updater.ploss, param)
  ρ1 = updater.ρ1
  mij = ρ1 * state.m[i,j] + (1.0 - ρ1) * gradient
  state.m[i,j] = mij
  uij = max(updater.ρ2 * state.u[i,j], abs(gradient))
  state.u[i,j] = uij
  if i == 1 && j == 1
    state.ρ1t[1] *= ρ1
  end
  -updater.η * mij / ((uij + 1e-10) * (1.0 - state.ρ1t[1]))
end

# -------------------------------------------------------------

