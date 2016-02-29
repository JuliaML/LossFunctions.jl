
x = [-1,0,1]

@test value(IdentityActivation(), x)    == x
@test value(SigmoidActivation(), x)     == map(sigmoid, x)
@test value(TanhActivation(), x)        == map(tanh, x)
@test value(SoftsignActivation(), x)    == map(xi -> xi / (1.0 + abs(xi)), x)
@test value(ReLUActivation(), x)        == map(xi -> max(0, xi), x)
@test value(LReLUActivation(0.01), x)   == map(xi -> xi * (xi > 0 ? 1 : 0.01), x)
@test value(SoftmaxActivation(), x)     == (evec = exp(x); evec/sum(evec))
