
# ------------------------------------------------------------
# minimizeable.jl
# ------------------------------------------------------------

export
    MinimizableFunctor,
        DifferentiableFunctor,
            TwiceDifferentiableFunctor

abstract MinimizableFunctor

value(::MinimizableFunctor, x) = @_not_implemented()

abstract DifferentiableFunctor <: MinimizableFunctor

value(::DifferentiableFunctor, x) = @_not_implemented()
grad!(buffer, ::DifferentiableFunctor, x) = @_not_implemented()
value_grad!(buffer, ::DifferentiableFunctor, x) = @_not_implemented()

abstract TwiceDifferentiableFunctor <: DifferentiableFunctor

value(::TwiceDifferentiableFunctor, x) = @_not_implemented()
grad!(buffer, ::TwiceDifferentiableFunctor, x) = @_not_implemented()
value_grad!(buffer, ::TwiceDifferentiableFunctor, x) = @_not_implemented()
hess!(buffer, ::TwiceDifferentiableFunctor, x) = @_not_implemented()


# ------------------------------------------------------------
# optimize.jl
# ------------------------------------------------------------

export
    AbstractOptimizer,
    AbstractSolver,
    optimize

"For anything that optimize an objective function"
abstract AbstractOptimizer

"For anything that can fit a model given some data"
abstract AbstractSolver

optimize(f::MinimizableFunctor, x::AbstractArray, o::AbstractOptimizer; nargs...) = @_not_implemented


# ------------------------------------------------------------
# parameter updater/state
# ------------------------------------------------------------

export
    ParameterUpdater,
    ParameterUpdaterState,

    ParameterUpdaters,
    ParameterUpdaterStates,

    get_state_type,
    param_change!

"""
An algorithm for gradient-based parameter updates. 

Use `param_change!` along with a `ParameterUpdaterState`.

Agorithms implemented:
    SGDUpdater (SGD with momentum)
    AdagradUpdater
    AdadeltaUpdater
    AdamUpdater
    AdaMaxUpdater
"""
abstract ParameterUpdater

"The current state of a gradient-based parameter update algorithm."
abstract ParameterUpdaterState

include("paramupdater.jl")

@autocomplete ParameterUpdaters export
    SGDUpdater,
    AdagradUpdater,
    AdadeltaUpdater,
    AdamUpdater,
    AdaMaxUpdater

@autocomplete ParameterUpdaterStates export
    SGDState,
    AdagradState,
    AdadeltaState,
    AdamState,
    AdaMaxState
    

# ------------------------------------------------------------
# sensitivity / error responsibility calculations
# ------------------------------------------------------------

include("sensitivity.jl")

export
    sensitivity,
    sensitivity!

