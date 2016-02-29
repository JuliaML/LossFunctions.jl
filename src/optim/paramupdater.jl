
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
    AdaMaxState

    # param_state

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

# immutable SGDState{T <: AbstractVecOrMat} <: ParameterUpdaterState
#     lastChanges::T
# end
# SGDState(dims::Integer...) = SGDState(zeros(dims...))
# param_state(updater::SGDUpdater, dims::Integer...) = SGDState(dims...)

type SGDState <: ParameterUpdaterState
    lastChange::Float64
end
SGDState() = SGDState(0.0)

"Calculate the amount `Δwᵢⱼ` to adjust the parameter."
function param_change!(state::SGDState, updater::SGDUpdater, gradient::Real, param::Real)
    gradient += deriv(updater.ploss, param)
    state.lastChange = -updater.η * gradient + updater.μ * state.lastChange
end

# -------------------------------------------------------------

"Adaptive Gradient"
type AdagradUpdater{PLOSS <: ParameterLoss} <: ParameterUpdater
    ε::Float64  # try 0.01?
    η::Float64 # base learning rate (numerator)
    ploss::PLOSS
end
AdagradUpdater(; ε=1e-8, η=1.0, ploss=NoParameterLoss()) = AdagradUpdater(ε, η, ploss)

# immutable AdagradState{T <: AbstractVecOrMat} <: ParameterUpdaterState
#     G::T
# end
# AdagradState(dims::Integer...) = AdagradState(zeros(dims...))
# param_state(updater::AdagradUpdater, dims::Integer...) = AdagradState(dims...)

type AdagradState <: ParameterUpdaterState
    G::Float64
end
AdagradState() = AdagradState(0.0)

function param_change!(state::AdagradState, updater::AdagradUpdater, gradient::Real, param::Real)
    gradient += deriv(updater.ploss, param)
    state.G += gradient^2
    η = updater.η / sqrt(updater.ε + state.G)
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


# immutable AdadeltaState{T <: AbstractVecOrMat} <: ParameterUpdaterState
#     dMean::T
#     GMean::T
# end
# AdadeltaState(dims::Integer...) = AdadeltaState(zeros(dims...), zeros(dims...))
# param_state(updater::AdadeltaUpdater, dims::Integer...) = AdadeltaState(dims...)

type AdadeltaState <: ParameterUpdaterState
    dMean::Float64
    GMean::Float64
end
AdadeltaState() = AdadeltaState(0.0, 0.0)

function param_change!(state::AdadeltaState, updater::AdadeltaUpdater, gradient::Real, param::Real)
    gradient += deriv(updater.ploss, param)
    ε, ρ = updater.ε, updater.ρ

    # average g²
    state.GMean = ρ * state.GMean + (1.0 - ρ) * gradient^2

    # compute learning rate from previous average dw² and current average g²
    η = updater.η * sqrt(state.dMean + ε) / sqrt(state.GMean + ε)

    # compute change and update average dw²
    dij = -η * gradient
    state.dMean = ρ * state.dMean + (1.0 - ρ) * dij^2
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

# type AdamState{T <: AbstractVecOrMat} <: ParameterUpdaterState
#     m::T # average first moment
#     v::T # average second moment
#     ρ1t::Float64  # β₁ᵗ from the paper... t-th power of β₁
#     ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
# end
# AdamState(dims::Integer...) = AdamState(zeros(dims...), zeros(dims...), 1.0, 1.0)
# param_state(updater::AdamUpdater, dims::Integer...) = AdamState(dims...)

type AdamState <: ParameterUpdaterState
    m::Float64      # average first moment
    v::Float64      # average second moment
    ρ1t::Float64    # β₁ᵗ from the paper... t-th power of β₁
    ρ2t::Float64    # β₂ᵗ from the paper... t-th power of β₂
end
AdamState() = AdamState(0.0, 0.0, 1.0, 1.0)

function param_change!(state::AdamState, updater::AdamUpdater, gradient::Real, param::Real)
    gradient += deriv(updater.ploss, param)
    ρ1, ρ2 = updater.ρ1, updater.ρ2
    state.m = ρ1 * state.m + (1.0 - ρ1) * gradient
    state.v = ρ2 * state.v + (1.0 - ρ2) * gradient^2
    state.ρ1t *= updater.ρ1
    state.ρ2t *= updater.ρ2
    ηt = updater.η * (sqrt(1.0 - state.ρ2t) / (1.0 - state.ρ1t))
    -ηt * state.m / (sqrt(state.v + updater.ε))
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

# immutable AdaMaxState{T <: AbstractVecOrMat} <: ParameterUpdaterState
#     m::T # average first moment
#     u::T # average second moment
#     ρ1t::Vector{Float64}  # β₁ᵗ from the paper... t-th power of β₁
#     # ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
# end
# AdaMaxState(dims::Integer...) = AdaMaxState(zeros(dims...), zeros(dims...), [1.0])
# param_state(updater::AdaMaxUpdater, dims::Integer...) = AdaMaxState(dims...)

type AdaMaxState <: ParameterUpdaterState
    m::Float64      # average first moment
    u::Float64      # average second moment
    p1t::Float64    # β₁ᵗ from the paper... t-th power of β₁
    # ρ2t::Float64  # β₂ᵗ from the paper... t-th power of β₂
end
AdaMaxState() = AdaMaxState(0.0, 0.0, 1.0)

function param_change!(state::AdaMaxState, updater::AdaMaxUpdater, gradient::Real, param::Real)
    gradient += deriv(updater.ploss, param)
    ρ1 = updater.ρ1
    mij = ρ1 * state.m + (1.0 - ρ1) * gradient
    state.m = mij
    uij = max(updater.ρ2 * state.u, abs(gradient))
    state.u = uij
    state.ρ1t[1] *= ρ1
    -updater.η * mij / ((uij + 1e-10) * (1.0 - state.ρ1t[1]))
end

# -------------------------------------------------------------

# Constructors for updater states

get_state_type(::SGDUpdater)        = SGDState
get_state_type(::AdagradUpdater)    = AdagradState
get_state_type(::AdadeltaUpdater)   = AdadeltaState
get_state_type(::AdamUpdater)       = AdamState
get_state_type(::AdaMaxUpdater)     = AdaMaxState

function ParameterUpdaterState(updater::ParameterUpdater, dims::Integer...)
    T = get_state_type(updater)
    states = Array(T, dims...)
    for i in eachindex(states)
        states[i] = T()
    end
    states
end

function ParameterUpdaterState(updater::ParameterUpdater)
    get_state_type(updater)()
end


# -------------------------------------------------------------

