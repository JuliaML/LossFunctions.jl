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
