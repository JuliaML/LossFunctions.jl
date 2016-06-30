using LearnBase
using Losses
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
    "tst_loss.jl"
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
