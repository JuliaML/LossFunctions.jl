
function test_margin_deriv(l::SupervisedLoss, t_vec)
  for y in [-1., 1], t in t_vec
    if isdifferentiable(l, y*t)
      d_dual = y * epsilon(value(l, dual(y, 0), dual(t, 1)))
      d_comp = deriv(l, y, t)
      @test abs(d_dual - d_comp) < 1e-14
      val = value(l, y, t)
      val2, d_comp2 = value_deriv(l, y, t)
      val3, d_comp3 = value_deriv_fun(l)(y, t)
      val4, d_comp4 = value_deriv(l, y * t)
      @test_approx_eq val val2
      @test_approx_eq val val3
      @test_approx_eq val val4
      @test_approx_eq val value(l, y*t)
      @test_approx_eq val value(l, [2,3], y, t)
      @test_approx_eq val value_fun(l)(y, t)
      @test_approx_eq val repr_fun(l)(y*t)
      @test_approx_eq val l(y*t)
      @test_approx_eq d_comp d_comp2
      @test_approx_eq d_comp d_comp3
      @test_approx_eq d_comp d_comp4
      @test_approx_eq d_comp deriv(l, y*t)
      @test_approx_eq d_comp deriv(l, [2,3], y, t)
      @test_approx_eq d_comp deriv_fun(l)(y, t)
      @test_approx_eq d_comp repr_deriv_fun(l)(y*t)
      @test_approx_eq d_comp l'(y*t)
    else
      print("(no $(y)*$(t)) ")
    end
  end
  println("ok")
end

function test_margin_deriv2(l::SupervisedLoss, t_vec)
  for y in [-1., 1], t in t_vec
    if istwicedifferentiable(l, y*t)
      d2_dual = y * epsilon(deriv(l, dual(y, 0), dual(t, 1)))
      d2_comp = deriv2(l, y, t)
      @test abs(d2_dual - d2_comp) < 1e-14
      @test_approx_eq d2_comp deriv2(l, y*t)
      @test_approx_eq d2_comp deriv2(l, [2,3], y, t)
      @test_approx_eq d2_comp deriv2_fun(l)(y, t)
    else
      print("(no $(y)*$(t)) ")
    end
  end
  println("ok")
end

# ==========================================================================

margin_losses = [LogitMarginLoss(), L1HingeLoss(), L2HingeLoss(),
                 L2SmoothedHingeLoss(.5), L2SmoothedHingeLoss(1), L2SmoothedHingeLoss(2)]

msg("Test first derivatives of margin-based losses")

for loss in margin_losses
  msg2("$(loss): ")
  test_margin_deriv(loss, -10:0.1:10)
end

msg("Test second derivatives of margin-based losses")

for loss in margin_losses
  msg2("$(loss): ")
  test_margin_deriv2(loss, -10:0.1:10)
end

# ==========================================================================
