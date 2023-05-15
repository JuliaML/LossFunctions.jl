function test_vector_value(l, o, t)
  ref = [l(o[i], t[i]) for i in 1:length(o)]
  v(l, o, t) = l.(o, t)
  @test @inferred(v(l, o, t)) == ref
  n = length(ref)
  s = sum(ref)
  @test @inferred(sum(l, o, t)) ≈ s
  @test @inferred(mean(l, o, t)) ≈ s / n
  @test @inferred(sum(l, o, t, ones(n), normalize=false)) ≈ s
  @test @inferred(sum(l, o, t, ones(n), normalize=true)) ≈ s / n
  @test @inferred(mean(l, o, t, ones(n), normalize=false)) ≈ s / n
  @test @inferred(mean(l, o, t, ones(n), normalize=true)) ≈ (s / n) / n
end

function test_vector_deriv(l, o, t)
  ref = [deriv(l, o[i], t[i]) for i in 1:length(o)]
  d(l, o, t) = deriv.(l, o, t)
  @test @inferred(d(l, o, t)) == ref
end

function test_vector_deriv2(l, o, t)
  ref = [deriv2(l, o[i], t[i]) for i in 1:length(o)]
  d(l, o, t) = deriv2.(l, o, t)
  @test @inferred(d(l, o, t)) == ref
end

@testset "Vectorized API" begin
  for T in (Float32, Float64)
    for O in (Float32, Float64)
      @testset "Margin-based $T -> $O" begin
        for (targets, outputs) in ((rand(T[-1, 1], 4), (rand(O, 4) .- O(0.5)) .* O(20)),)
          for loss in (LogitMarginLoss(), ModifiedHuberLoss(), L1HingeLoss(), SigmoidLoss())
            @testset "$(loss): " begin
              test_vector_value(loss, outputs, targets)
              test_vector_deriv(loss, outputs, targets)
              test_vector_deriv2(loss, outputs, targets)
            end
          end
        end
      end
      @testset "Distance-based $T -> $O" begin
        for (targets, outputs) in (((rand(T, 4) .- T(0.5)) .* T(20), (rand(O, 4) .- O(0.5)) .* O(20)),)
          for loss in (QuantileLoss(0.75), L2DistLoss(), EpsilonInsLoss(1))
            @testset "$(loss): " begin
              test_vector_value(loss, outputs, targets)
              test_vector_deriv(loss, outputs, targets)
              test_vector_deriv2(loss, outputs, targets)
            end
          end
        end
      end
    end
  end
end

@testset "Aggregation with categorical values" begin
  c = categorical(["Foo", "Bar", "Baz", "Foo"])
  l = MisclassLoss()
  @test sum(l, c, reverse(c)) == 2.0
  @test mean(l, c, reverse(c)) == 0.5
  @test sum(l, c, reverse(c), 2 * ones(4), normalize=false) == 4.0
  @test mean(l, c, reverse(c), 2 * ones(4), normalize=false) == 1.0
  @test mean(l, c, reverse(c), 2 * ones(4), normalize=true) == 0.125
end
