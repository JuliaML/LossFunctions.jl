
function test_value(l::SupervisedLoss, f::Function, y_vec, t_vec)
  msg2("$(l): ")
  for y in y_vec, t in t_vec
    @test abs(l(y, t) - f(y, t)) < 1e-10
  end
  println("ok")
end

function test_deriv(l::MarginBasedLoss, t_vec)
  msg2("$(l): ")
  for y in [-1., 1], t in t_vec
    if isdifferentiable(l, y*t)
      d_dual = epsilon(value(l, dual(y, 0), dual(t, 1)))
      d_comp = deriv(l, y, t)
      @test abs(d_dual - d_comp) < 1e-10
      val = value(l, y, t)
      val2, d_comp2 = value_deriv(l, y, t)
      val3, d_comp3 = value_deriv_fun(l)(y, t)
      val4, d_comp4 = value_deriv(l, y * t)
      @test_approx_eq val val2
      @test_approx_eq val val3
      @test_approx_eq val val4
      @test_approx_eq val value(l, y, t)
      @test_approx_eq val value(l, y*t)
      @test_approx_eq val value(l, [2,3], y, t)
      @test_approx_eq val value_fun(l)(y, t)
      @test_approx_eq val value_fun(l)(y*t)
      @test_approx_eq val repr_fun(l)(y*t)
      @test_approx_eq val l(y*t)
      @test_approx_eq d_comp d_comp2
      @test_approx_eq d_comp d_comp3
      @test_approx_eq d_comp y*d_comp4
      @test_approx_eq d_comp y*deriv(l, y*t)
      @test_approx_eq d_comp deriv(l, [2,3], y, t)
      @test_approx_eq d_comp deriv_fun(l)(y, t)
      @test_approx_eq d_comp y*deriv_fun(l)(y*t)
      @test_approx_eq d_comp y*repr_deriv_fun(l)(y*t)
      @test_approx_eq d_comp l'(y,t)
      @test_approx_eq d_comp y*l'(y*t)
    else
      # y*t == 1 ? print(".") : print("(no $(y)*$(t)) ")
      print(".")
    end
  end
  println("ok")
end

function test_deriv(l::DistanceBasedLoss, t_vec)
  msg2("$(l): ")
  for y in -20:.2:20, t in t_vec
    if isdifferentiable(l, t-y)
      d_dual = epsilon(value(l, dual(t-y, 1)))
      d_comp = deriv(l, y, t)
      @test abs(d_dual - d_comp) < 1e-10
      val = value(l, y, t)
      val2, d_comp2 = value_deriv(l, y, t)
      val3, d_comp3 = value_deriv_fun(l)(y, t)
      val4, d_comp4 = value_deriv(l, t-y)
      @test_approx_eq val val2
      @test_approx_eq val val3
      @test_approx_eq val val4
      @test_approx_eq val value(l, y, t)
      @test_approx_eq val value(l, t-y)
      @test_approx_eq val value(l, [2,3], y, t)
      @test_approx_eq val value_fun(l)(y, t)
      @test_approx_eq val value_fun(l)(t-y)
      @test_approx_eq val repr_fun(l)(t-y)
      @test_approx_eq val l(t-y)
      @test_approx_eq d_comp d_comp2
      @test_approx_eq d_comp d_comp3
      @test_approx_eq d_comp d_comp4
      @test_approx_eq d_comp deriv(l, t-y)
      @test_approx_eq d_comp deriv(l, [2,3], y, t)
      @test_approx_eq d_comp deriv_fun(l)(y, t)
      @test_approx_eq d_comp deriv_fun(l)(t-y)
      @test_approx_eq d_comp repr_deriv_fun(l)(t-y)
      @test_approx_eq d_comp l'(t-y)
    else
      # y-t == 0 ? print(".") : print("$(y-t) ")
      print(".")
    end
  end
  println("ok")
end

function test_deriv2(l::MarginBasedLoss, t_vec)
  msg2("$(l): ")
  for y in [-1., 1], t in t_vec
    if istwicedifferentiable(l, y*t)
      d2_dual = epsilon(deriv(l, dual(y, 0), dual(t, 1)))
      d2_comp = deriv2(l, y, t)
      @test abs(d2_dual - d2_comp) < 1e-10
      @test_approx_eq d2_comp deriv2(l, y, t)
      @test_approx_eq d2_comp deriv2(l, y*t)
      @test_approx_eq d2_comp deriv2(l, [2,3], y, t)
      @test_approx_eq d2_comp deriv2_fun(l)(y, t)
      @test_approx_eq d2_comp deriv2_fun(l)(y*t)
    else
      # y*t == 1 ? print(".") : print("(no $(y)*$(t)) ")
      print(".")
    end
  end
  println("ok")
end

function test_deriv2(l::DistanceBasedLoss, t_vec)
  msg2("$(l): ")
  for y in -20:.2:20, t in t_vec
    if istwicedifferentiable(l, t-y)
      d2_dual = epsilon(deriv(l, dual(t-y, 1)))
      d2_comp = deriv2(l, y, t)
      @test abs(d2_dual - d2_comp) < 1e-10
      @test_approx_eq d2_comp deriv2(l, y, t)
      @test_approx_eq d2_comp deriv2(l, t-y)
      @test_approx_eq d2_comp deriv2(l, [2,3], y, t)
      @test_approx_eq d2_comp deriv2_fun(l)(y, t)
      @test_approx_eq d2_comp deriv2_fun(l)(t-y)
    else
      # y-t == 0 ? print(".") : print("$(y-t) ")
      print(".")
    end
  end
  println("ok")
end

# ==========================================================================

msg("Test margin-based loss against reference function")

_hingeloss(y, t) = max(0, 1 - y.*t)
test_value(HingeLoss(), _hingeloss, [-1.,1], -10:0.1:10)

_l2hingeloss(y, t) = max(0, 1 - y.*t)^2
test_value(L2HingeLoss(), _l2hingeloss, [-1.,1], -10:0.1:10)

_perceptronloss(y, t) = max(0, -y.*t)
test_value(PerceptronLoss(), _perceptronloss, [-1.,1], -10:0.1:10)

_logitmarginloss(y, t) = log(1 + exp(-y.*t))
test_value(LogitMarginLoss(), _logitmarginloss, [-1.,1], -10:0.1:10)

function _smoothedl1hingeloss(γ)
  function _value(y, t)
    if y.*t >= 1 - γ
      1/(2γ) * max(0, 1- y.*t)^2
    else
      1 - γ / 2 - y.*t
    end
  end
  _value
end
test_value(SmoothedL1HingeLoss(.5), _smoothedl1hingeloss(.5), [-1.,1], -10:0.1:10)
test_value(SmoothedL1HingeLoss(1), _smoothedl1hingeloss(1), [-1.,1], -10:0.1:10)
test_value(SmoothedL1HingeLoss(2), _smoothedl1hingeloss(2), [-1.,1], -10:0.1:10)

function _modhuberloss(y, t)
  if y.*t >= -1
    max(0, 1 - y.*t)^2
  else
    -4.*y.*t
  end
end
test_value(ModifiedHuberLoss(), _modhuberloss, [-1.,1], -10:0.1:10)

# ==========================================================================

msg("Test distance-based loss against reference function")

_l1distloss(y, t) = abs(t - y)
test_value(L1DistLoss(), _l1distloss, -20:.2:20, -30:0.5:30)

_l2distloss(y, t) = (t - y)^2
test_value(L2DistLoss(), _l2distloss, -20:.2:20, -30:0.5:30)

_lp15distloss(y, t) = abs(t - y)^(1.5)
test_value(LPDistLoss(1.5), _lp15distloss, -20:.2:20, -30:0.5:30)

function _l1epsinsloss(ɛ)
  _value(y, t) = max(0, abs(t - y) - ɛ)
  _value
end
test_value(EpsilonInsLoss(.5), _l1epsinsloss(0.5), -20:.2:20, -30:0.5:30)
test_value(EpsilonInsLoss(1), _l1epsinsloss(1), -20:.2:20, -30:0.5:30)
test_value(EpsilonInsLoss(1.5), _l1epsinsloss(1.5), -20:.2:20, -30:0.5:30)

function _l2epsinsloss(ɛ)
  _value(y, t) = max(0, abs(t - y) - ɛ)^2
  _value
end
test_value(L2EpsilonInsLoss(.5), _l2epsinsloss(0.5), -20:.2:20, -30:0.5:30)
test_value(L2EpsilonInsLoss(1), _l2epsinsloss(1), -20:.2:20, -30:0.5:30)
test_value(L2EpsilonInsLoss(1.5), _l2epsinsloss(1.5), -20:.2:20, -30:0.5:30)

_logitdistloss(y, t) = -log((4*exp(t-y))/(1+exp(t-y))^2)
test_value(LogitDistLoss(), _logitdistloss, -20:.2:20, -30:0.5:30)

# ==========================================================================

msg("Test other loss against reference function")

_crossentropyloss(y, t) = -y*log(t) - (1-y)*log(1-t)
test_value(CrossentropyLoss(), _crossentropyloss, 0:0.01:1, 0.01:0.01:0.99)

_zerooneloss(y, t) = sign(y*t) < 0 ? 1 : 0
test_value(ZeroOneLoss(), _zerooneloss, [-1.,1], -10:0.1:10)

# ==========================================================================

margin_losses = [LogitMarginLoss(), L1HingeLoss(), L2HingeLoss(), PerceptronLoss(),
                 SmoothedL1HingeLoss(.5), SmoothedL1HingeLoss(1), SmoothedL1HingeLoss(2),
                 ModifiedHuberLoss()]

msg("Test first derivatives of margin-based losses")

for loss in margin_losses
  test_deriv(loss, -10:0.1:10)
end

msg("Test second derivatives of margin-based losses")

for loss in margin_losses
  test_deriv2(loss, -10:0.1:10)
end

# ==========================================================================

distance_losses = [L2DistLoss(), LPDistLoss(2.0), L1DistLoss(), LPDistLoss(1.0),
                   LPDistLoss(0.5), LPDistLoss(1.5), LPDistLoss(3),
                   LogitDistLoss(), L1EpsilonInsLoss(0.5), EpsilonInsLoss(1.5),
                   L2EpsilonInsLoss(0.5), L2EpsilonInsLoss(1.5)]

msg("Test first derivatives of distance-based losses")

for loss in distance_losses
  test_deriv(loss, -30:0.5:30)
end

msg("Test second derivatives of distance-based losses")

for loss in distance_losses
  test_deriv2(loss, -30:0.5:30)
end

