using LossFunctions
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

# ==========================================================================
# Specify tests

tests = [
  "tst_loss.jl"
]

for t in tests
  println("[->] $t")
  include(t)
  println("[OK] $t")
  println("====================================================================")
end
