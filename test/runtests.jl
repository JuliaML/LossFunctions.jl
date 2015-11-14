using LearnBase
using LearnBase.LossFunctions
using LearnBase.Penalties
using DualNumbers
using Base.Test

function msg(args...; newline = true)
    print("   --> ", args...)
    newline && println()
end

function msg2(args...; newline = false)
    print("       - ", args...)
    newline && println()
end

tests = [
    "tst_loss.jl"
    "tst_classencoding.jl"
    "tst_penalty.jl"
    "tst_prediction.jl"
]

perf = [
    #"bm_datasource.jl"
]

for t in tests
    println("[->] $t")
    include(t)
    println("[OK] $t")
    println("====================================================================")
end

for p in perf
    println("[->] $p")
    include(p)
    println("[OK] $p")
    println("====================================================================")
end
