using LearnBase
using LossFunctions
using DualNumbers
using Base.Test

tests = [
    "tst_loss.jl",
    "tst_api.jl",
    "tst_properties.jl"
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
