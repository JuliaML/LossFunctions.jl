
# ---------------------------------------------------------------------------------
# Warning: This directory will likely get an overhaul, so please don't use it yet.
# ---------------------------------------------------------------------------------

export
    sigmoid,
    Predictor,
        LinearPredictor,
        SigmoidPredictor,

    EmpiricalRisk,
        EmpiricalRiskClassifier,
        EmpiricalRiskRegressor,
    RiskFunctional

include("prediction.jl")
include("empiricalrisk.jl")
include("riskfunc.jl")

