"For anything that optimize an objective function"
abstract AbstractOptimizer

"For anything that can fit a model given some data"
abstract AbstractSolver

optimize(f::MinimizableFunctor, x::AbstractArray, o::AbstractOptimizer; nargs...) = @_not_implemented
