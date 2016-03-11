
x = [-1,0,1]

@test value(IdentityMapping(), x)    == x
@test value(SigmoidMapping(), x)     == map(sigmoid, x)
@test value(TanhMapping(), x)        == map(tanh, x)
@test value(SoftsignMapping(), x)    == map(xi -> xi / (1.0 + abs(xi)), x)
@test value(ReLUMapping(), x)        == map(xi -> max(0, xi), x)
@test value(LReLUMapping(0.01), x)   == map(xi -> xi * (xi >= 0 ? 1 : 0.01), x)
@test value(SoftmaxMapping(), x)     == (evec = exp(x); evec/sum(evec))

@test deriv(IdentityMapping(), x)    == ones(x)
@test deriv(SigmoidMapping(), x)     == (y = map(sigmoid, x); y .* (1-y))
@test deriv(TanhMapping(), x)        == 1 - map(tanh,x) .^ 2
@test deriv(SoftsignMapping(), x)    == 1 ./ (1 + abs(x)) .^ 2
@test deriv(ReLUMapping(), x)        == map(xi -> xi >= 0 ? 1 : 0, x)
@test deriv(LReLUMapping(0.01), x)   == map(xi -> xi >= 0 ? 1 : 0.01, x)
@test_throws ArgumentError deriv(SoftmaxMapping(), x)
