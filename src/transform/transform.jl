
abstract Transformation

export
    Transformation,
        Mapping

# ------------------------------------
# Mapping Functions
# ------------------------------------

include("mapping.jl")

@autocomplete Mappings export
    IdentityMapping,
    SigmoidMapping,
    TanhMapping,
    SoftsignMapping,
    ReLUMapping,
    LReLUMapping,
    SoftmaxMapping


# ------------------------------------
# ------------------------------------
