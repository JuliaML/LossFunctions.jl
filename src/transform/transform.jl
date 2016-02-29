
abstract Transformation

export
    Transformation,
        Activation

# ------------------------------------
# Activation Functions
# ------------------------------------

include("activation.jl")

@autocomplete Activations export
    IdentityActivation,
    SigmoidActivation,
    TanhActivation,
    SoftsignActivation,
    ReLUActivation,
    LReLUActivation,
    SoftmaxActivation


# ------------------------------------
# ------------------------------------
