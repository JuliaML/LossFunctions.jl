using LearnBase
using MLModels
using MLDataUtils
using UnicodePlots
using DualNumbers

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

tests = [
    "tst_mapping.jl"
    "tst_loss.jl"
    "tst_penalty.jl"
    "tst_prediction.jl"
    "tst_empiricalrisk.jl"
    "tst_riskfunc.jl"
]

perf = [
    #"bm_datasource.jl"
]

# for deterministic testing
srand(1234)

for t in tests
    @testset "$t" begin
        include(t)
    end
end

for p in perf
    @testset "$p" begin
        include(p)
    end
end
