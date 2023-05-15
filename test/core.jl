function test_value_typestable(l::SupervisedLoss)
  @testset "$(l): " begin
    for o in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-0.5), Float32(0.5))
      for t in (-2, 2, Int32(-1), Int32(1), -0.5, 0.5, Float32(-1), Float32(1))
        # check inference
        @inferred deriv(l, o, t)
        @inferred deriv2(l, o, t)

        # get expected return type
        T = promote_type(typeof(o), typeof(t))

        # test basic loss
        val = @inferred l(o, t)
        @test typeof(val) <: T

        # test scaled version of loss
        @test typeof((T(2) * l)(o, t)) <: T
      end
    end
  end
end

function test_value_float32_preserving(l::SupervisedLoss)
  @testset "$(l): " begin
    for o in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-0.5), Float32(0.5))
      for t in (-2, 2, Int32(-1), Int32(1), -0.5, 0.5, Float32(-1), Float32(1))
        # check inference
        @inferred deriv(l, o, t)
        @inferred deriv2(l, o, t)

        val = @inferred l(o, t)
        T = promote_type(typeof(o), typeof(t))
        if !(T <: AbstractFloat)
          # cast Integers to a float
          # (whether its Float32 or Float64 depends on the loss...)
          @test (typeof(val) <: AbstractFloat)
        elseif T <: Float32
          # preserve Float32
          @test (typeof(val) <: Float32)
        else
          @test (typeof(val) <: Float64)
        end
      end
    end
  end
end

function test_value_float64_forcing(l::SupervisedLoss)
  @testset "$(l): " begin
    for o in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-0.5), Float32(0.5))
      for t in (-2, 2, Int32(-1), Int32(1), -0.5, 0.5, Float32(-1), Float32(1))
        # check inference
        @inferred deriv(l, o, t)
        @inferred deriv2(l, o, t)

        val = @inferred l(o, t)
        @test (typeof(val) <: Float64)
      end
    end
  end
end

function test_value(l::SupervisedLoss, f::Function, o_vec, t_vec)
  @testset "$(l): " begin
    for o in o_vec, t in t_vec
      @test abs(l(o, t) - f(o, t)) < 1e-10
    end
  end
end

function test_deriv(l::MarginLoss, o_vec)
  @testset "$(l): " begin
    for o in o_vec, t in [-1.0, 1.0]
      if isdifferentiable(l, o * t)
        d_dual = epsilon(l(dual(o, one(o)), dual(t, zero(t))))
        d_comp = @inferred deriv(l, o, t)
        @test abs(d_dual - d_comp) < 1e-10
        val = @inferred l(o, t)
        @test val ≈ l(o, t)
        @test val ≈ l(o * t)
        @test d_comp ≈ t * deriv(l, o * t)
      end
    end
  end
end

function test_deriv(l::DistanceLoss, o_vec)
  @testset "$(l): " begin
    for o in o_vec, t in -10:0.2:10
      if isdifferentiable(l, o - t)
        d_dual = epsilon(l(dual(o - t, one(o - t))))
        d_comp = @inferred deriv(l, o, t)
        @test abs(d_dual - d_comp) < 1e-10
        val = @inferred l(o, t)
        @test val ≈ l(o, t)
        @test val ≈ l(o - t)
        @test d_comp ≈ deriv(l, o - t)
      end
    end
  end
end

function test_deriv(l::SupervisedLoss, o_vec, t_vec)
  @testset "$(l): " begin
    for o in o_vec, t in t_vec
      if isdifferentiable(l, o, t)
        d_dual = epsilon(l(dual(o, one(o)), dual(t, zero(t))))
        d_comp = @inferred deriv(l, o, t)
        @test abs(d_dual - d_comp) < 1e-10
        val = @inferred l(o, t)
        @test val ≈ l(o, t)
        @test d_comp ≈ deriv(l, o, t)
      end
    end
  end
end

function test_deriv2(l::MarginLoss, o_vec)
  @testset "$(l): " begin
    for o in o_vec, t in [-1.0, 1]
      if istwicedifferentiable(l, o * t) && isdifferentiable(l, o * t)
        d2_dual = epsilon(deriv(l, dual(o, one(o)), dual(t, zero(t))))
        d2_comp = @inferred deriv2(l, o, t)
        @test abs(d2_dual - d2_comp) < 1e-10
        @test d2_comp ≈ @inferred deriv2(l, o, t)
        @test d2_comp ≈ @inferred deriv2(l, o * t)
      end
    end
  end
end

function test_deriv2(l::DistanceLoss, o_vec)
  @testset "$(l): " begin
    for o in o_vec, t in -10:0.2:10
      if istwicedifferentiable(l, o - t) && isdifferentiable(l, o - t)
        d2_dual = epsilon(deriv(l, dual(o - t, one(o - t))))
        d2_comp = @inferred deriv2(l, o, t)
        @test abs(d2_dual - d2_comp) < 1e-10
        @test d2_comp ≈ @inferred deriv2(l, o, t)
        @test d2_comp ≈ @inferred deriv2(l, o - t)
      end
    end
  end
end

function test_deriv2(l::SupervisedLoss, o_vec, t_vec)
  @testset "$(l): " begin
    for o in o_vec, t in t_vec
      if istwicedifferentiable(l, o, t) && isdifferentiable(l, o, t)
        d2_dual = epsilon(deriv(l, dual(o, one(o)), dual(t, zero(t))))
        d2_comp = @inferred deriv2(l, o, t)
        @test abs(d2_dual - d2_comp) < 1e-10
        @test d2_comp ≈ @inferred deriv2(l, o, t)
      end
    end
  end
end

function test_scaledloss(l::SupervisedLoss, o_vec, t_vec)
  @testset "Scaling for $(l): " begin
    for λ in (2.0, 2)
      sl = ScaledLoss(l, λ)
      @test typeof(sl) <: ScaledLoss{typeof(l),λ}
      @test 3 * sl == @inferred(ScaledLoss(sl, Val(3)))
      @test (λ * 3) * l == @inferred(ScaledLoss(sl, Val(3)))
      @test sl == @inferred(ScaledLoss(l, Val(λ)))
      @test sl == λ * l
      @test sl == @inferred(Val(λ) * l)
      for o in o_vec, t in t_vec
        @test @inferred(sl(o, t)) == λ * l(o, t)
        @test @inferred(deriv(sl, o, t)) == λ * deriv(l, o, t)
        @test @inferred(deriv2(sl, o, t)) == λ * deriv2(l, o, t)
      end
    end
  end
end

function test_weightedloss(l::MarginLoss, o_vec, t_vec)
  @testset "Weighted version for $(l): " begin
    for w in (0.0, 0.2, 0.7, 1.0)
      wl = WeightedMarginLoss(l, w)
      @test typeof(wl) <: WeightedMarginLoss{typeof(l),w}
      @test WeightedMarginLoss(l, w * 0.1) == WeightedMarginLoss(wl, 0.1)
      for o in o_vec, t in t_vec
        if t == 1
          @test wl(o, t) == w * l(o, t)
          @test deriv(wl, o, t) == w * deriv(l, o, t)
          @test deriv2(wl, o, t) == w * deriv2(l, o, t)
        else
          @test wl(o, t) == (1 - w) * l(o, t)
          @test deriv(wl, o, t) == (1 - w) * deriv(l, o, t)
          @test deriv2(wl, o, t) == (1 - w) * deriv2(l, o, t)
        end
      end
    end
  end
end

# ====================================================================

@testset "Test typealias" begin
  @test L1DistLoss === LPDistLoss{1}
  @test L2DistLoss === LPDistLoss{2}
  @test HingeLoss === L1HingeLoss
  @test EpsilonInsLoss === L1EpsilonInsLoss
end

@testset "Test typestable supervised loss for type stability" begin
  for loss in [
    L1HingeLoss(),
    L2HingeLoss(),
    ModifiedHuberLoss(),
    PerceptronLoss(),
    LPDistLoss(1),
    LPDistLoss(2),
    LPDistLoss(3),
    L2MarginLoss()
  ]
    test_value_typestable(loss)
    # TODO: add ZeroOneLoss after scaling works...
  end
end

@testset "Test float-forcing supervised loss for type stability" begin
  # Losses that should always return Float64
  for loss in [
    SmoothedL1HingeLoss(0.5),
    SmoothedL1HingeLoss(1),
    L1EpsilonInsLoss(0.5),
    L1EpsilonInsLoss(1),
    L2EpsilonInsLoss(0.5),
    L2EpsilonInsLoss(1),
    PeriodicLoss(1),
    PeriodicLoss(1.5),
    HuberLoss(1.0),
    QuantileLoss(0.8),
    DWDMarginLoss(0.5),
    DWDMarginLoss(1),
    DWDMarginLoss(2)
  ]
    test_value_float64_forcing(loss)
    test_value_float64_forcing(2.0 * loss)
  end
  test_value_float64_forcing(2.0 * LogitDistLoss())
  test_value_float64_forcing(2.0 * LogitMarginLoss())
  test_value_float64_forcing(2.0 * ExpLoss())
  test_value_float64_forcing(2.0 * SigmoidLoss())

  # Losses that should return an AbstractFloat, preserving type if possible
  for loss in [
    SmoothedL1HingeLoss(0.5f0),
    SmoothedL1HingeLoss(1.0f0),
    PeriodicLoss(1.0f0),
    PeriodicLoss(0.5f0),
    LogitDistLoss(),
    LogitMarginLoss(),
    ExpLoss(),
    SigmoidLoss(),
    L1EpsilonInsLoss(1.0f0),
    L1EpsilonInsLoss(0.5f0),
    L2EpsilonInsLoss(1.0f0),
    L2EpsilonInsLoss(0.5f0),
    HuberLoss(1.0f0),
    QuantileLoss(0.8f0),
    DWDMarginLoss(0.5f0)
  ]
    test_value_float32_preserving(loss)
    test_value_float32_preserving(2.0f0 * loss)
  end
end

@testset "Test margin-based loss against reference function" begin
  _zerooneloss(o, t) = sign(o * t) < 0 ? 1 : 0
  test_value(ZeroOneLoss(), _zerooneloss, -10:0.2:10, [-1.0, 1.0])

  _hingeloss(o, t) = max(0, 1 - o .* t)
  test_value(HingeLoss(), _hingeloss, -10:0.2:10, [-1.0, 1.0])

  _l2hingeloss(o, t) = max(0, 1 - o .* t)^2
  test_value(L2HingeLoss(), _l2hingeloss, -10:0.2:10, [-1.0, 1.0])

  _perceptronloss(o, t) = max(0, -o .* t)
  test_value(PerceptronLoss(), _perceptronloss, -10:0.2:10, [-1.0, 1.0])

  _logitmarginloss(o, t) = log(1 + exp(-o .* t))
  test_value(LogitMarginLoss(), _logitmarginloss, -10:0.2:10, [-1.0, 1.0])

  function _smoothedl1hingeloss(γ)
    function _value(o, t)
      if o .* t >= 1 - γ
        1 / (2γ) * max(0, 1 - o .* t)^2
      else
        1 - γ / 2 - o .* t
      end
    end
    _value
  end
  test_value(SmoothedL1HingeLoss(0.5), _smoothedl1hingeloss(0.5), -10:0.2:10, [-1.0, 1.0])
  test_value(SmoothedL1HingeLoss(1), _smoothedl1hingeloss(1), -10:0.2:10, [-1.0, 1.0])
  test_value(SmoothedL1HingeLoss(2), _smoothedl1hingeloss(2), -10:0.2:10, [-1.0, 1.0])

  function _modhuberloss(o, t)
    if o .* t >= -1
      max(0, 1 - o .* t)^2
    else
      -4 .* o .* t
    end
  end
  test_value(ModifiedHuberLoss(), _modhuberloss, -10:0.2:10, [-1.0, 1.0])

  _l2marginloss(o, t) = (1 - o .* t)^2
  test_value(L2MarginLoss(), _l2marginloss, -10:0.2:10, [-1.0, 1.0])

  _exploss(o, t) = exp(-o .* t)
  test_value(ExpLoss(), _exploss, -10:0.2:10, [-1.0, 1.0])

  _sigmoidloss(o, t) = (1 - tanh(o .* t))
  test_value(SigmoidLoss(), _sigmoidloss, -10:0.2:10, [-1.0, 1.0])

  function _dwdmarginloss(q)
    function _value(o, t)
      if o .* t <= q / (q + 1)
        convert(Float64, 1 - o .* t)
      else
        ((q^q) / (q + 1)^(q + 1)) / (o .* t)^q
      end
    end
    _value
  end
  test_value(DWDMarginLoss(0.5), _dwdmarginloss(0.5), -10:0.2:10, [-1.0, 1.0])
  test_value(DWDMarginLoss(1), _dwdmarginloss(1), -10:0.2:10, [-1.0, 1.0])
  test_value(DWDMarginLoss(2), _dwdmarginloss(2), -10:0.2:10, [-1.0, 1.0])
end

@testset "Test distance-based loss against reference function" begin
  or = range(-10, stop=20, length=10)
  tr = range(-30, stop=30, length=10)

  _l1distloss(o, t) = abs(t - o)
  test_value(L1DistLoss(), _l1distloss, or, tr)

  _l2distloss(o, t) = (t - o)^2
  test_value(L2DistLoss(), _l2distloss, or, tr)

  _lp15distloss(o, t) = abs(t - o)^(1.5)
  test_value(LPDistLoss(1.5), _lp15distloss, or, tr)

  function _periodicloss(c)
    (o, t) -> 1 - cos((o - t) * 2π / c)
  end
  test_value(PeriodicLoss(0.5), _periodicloss(0.5), or, tr)
  test_value(PeriodicLoss(1), _periodicloss(1), or, tr)
  test_value(PeriodicLoss(1.5), _periodicloss(1.5), or, tr)

  function _huberloss(d)
    (o, t) -> abs(o - t) < d ? (abs2(o - t) / 2) : (d * (abs(o - t) - (d / 2)))
  end
  test_value(HuberLoss(0.5), _huberloss(0.5), or, tr)
  test_value(HuberLoss(1), _huberloss(1), or, tr)
  test_value(HuberLoss(1.5), _huberloss(1.5), or, tr)

  function _l1epsinsloss(ɛ)
    (o, t) -> max(0, abs(t - o) - ɛ)
  end
  test_value(EpsilonInsLoss(0.5), _l1epsinsloss(0.5), or, tr)
  test_value(EpsilonInsLoss(1), _l1epsinsloss(1), or, tr)
  test_value(EpsilonInsLoss(1.5), _l1epsinsloss(1.5), or, tr)

  function _l2epsinsloss(ɛ)
    (o, t) -> max(0, abs(t - o) - ɛ)^2
  end
  test_value(L2EpsilonInsLoss(0.5), _l2epsinsloss(0.5), or, tr)
  test_value(L2EpsilonInsLoss(1), _l2epsinsloss(1), or, tr)
  test_value(L2EpsilonInsLoss(1.5), _l2epsinsloss(1.5), or, tr)

  _logitdistloss(o, t) = -log((4 * exp(t - o)) / (1 + exp(t - o))^2)
  test_value(LogitDistLoss(), _logitdistloss, or, tr)

  function _quantileloss(τ)
    (o, t) -> (o - t) * ((o - t > 0) - τ)
  end
  test_value(QuantileLoss(0.7), _quantileloss(0.7), or, tr)

  function _logcoshloss(o, t)
    log(cosh(o - t))
  end
  test_value(LogCoshLoss(), _logcoshloss, or, tr)
end

@testset "Test other loss" begin
  _misclassloss(o, t) = o == t ? 0 : 1
  test_value(MisclassLoss(), _misclassloss, vcat(1:5, 7:11), 1:10)

  _crossentropyloss(o, t) = -t * log(o) - (1 - t) * log(1 - o)
  test_value(CrossEntropyLoss(), _crossentropyloss, 0.01:0.01:0.99, 0:0.01:1)

  _poissonloss(o, t) = exp(o) - t * o
  test_value(PoissonLoss(), _poissonloss, range(0, stop=10, length=11), 0:10)
end

@testset "Test scaled loss" begin
  for loss in distance_losses
    test_scaledloss(loss, -10:0.5:10, -10:0.2:10)
  end

  for loss in margin_losses
    test_scaledloss(loss, -10:0.2:10, [-1.0, 1.0])
  end

  test_scaledloss(PoissonLoss(), range(0, stop=10, length=11), 0:10)
end

@testset "Test weighted loss" begin
  for loss in margin_losses
    test_weightedloss(loss, -10:0.2:10, [-1.0, 1.0])
  end
end

# --------------------------------------------------------------

@testset "Test first derivatives" begin
  for loss in distance_losses
    test_deriv(loss, -10:0.5:10)
  end

  for loss in margin_losses
    test_deriv(loss, -10:0.2:10)
  end

  test_deriv(PoissonLoss(), -10:0.2:10, 0:30)
  test_deriv(CrossEntropyLoss(), 0.01:0.01:0.99, 0:0.01:1)
end

# --------------------------------------------------------------

@testset "Test second derivatives" begin
  for loss in distance_losses
    test_deriv2(loss, -10:0.5:10)
  end

  for loss in margin_losses
    test_deriv2(loss, -10:0.2:10)
  end

  test_deriv2(PoissonLoss(), -10:0.2:10, 0:30)
  test_deriv2(CrossEntropyLoss(), 0.01:0.01:0.99, 0:0.01:1)
end

# --------------------------------------------------------------

@testset "Test losses with categorical values" begin
  c = categorical(["Foo", "Bar", "Baz", "Foo"])

  l = MisclassLoss()
  @test l(c[1], c[1]) == 0.0
  @test l(c[1], c[2]) == 1.0
  @test l.(c, reverse(c)) == [0.0, 1.0, 1.0, 0.0]

  l = MisclassLoss{Float32}()
  @test l(c[1], c[1]) isa Float32
  @test l.(c, c) isa Vector{Float32}
end
