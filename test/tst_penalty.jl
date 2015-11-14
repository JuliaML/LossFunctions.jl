function test_value(l::Penalty, f::Function)
    msg2("$(l): ")
    for i = 1:20
        w = randn(10)
        ref = value(l, w, length(w))::Float64
        val = f(w)::Float64
        @test abs(ref - val) < 1e-10
        ref = value(l, w, length(w)-1)::Float64
        val = f(w[1:(length(w)-1)])::Float64
        @test abs(ref - val) < 1e-10
    end
    println("ok")
end

# ==========================================================================

msg("Test penalties against reference function")

function _l1pen(λ)
  _value(w) = λ * sumabs(w)
  _value
end
test_value(L1Penalty(0.05), _l1pen(0.05))
test_value(L1Penalty(0.1), _l1pen(0.1))
test_value(L1Penalty(0.5), _l1pen(0.5))
test_value(L1Penalty(1), sumabs)

function _l2pen(λ)
  _value(w) = λ/2 * sumabs2(w)
  _value
end
test_value(L2Penalty(0.05), _l2pen(0.05))
test_value(L2Penalty(0.1), _l2pen(0.1))
test_value(L2Penalty(0.5), _l2pen(0.5))
test_value(L2Penalty(1), _l2pen(1))
test_value(L2Penalty(2), sumabs2)
