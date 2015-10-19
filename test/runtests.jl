using LearnBase
using Base.Test

function msg(args...)
  println("   --> ", args...)
end

tests = [
  "tst_classencoding.jl"
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
