
abstract Transformation

# ------------------------------------
# Activation Functions
# ------------------------------------

include("activation.jl")

@autocomplete Activations export
    Activation,
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    SoftsignActivation,
    ReLUActivation,
    LReLUActivation,
    SoftmaxActivation


# ------------------------------------
# ------------------------------------
