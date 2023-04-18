function test_vector_value(l, o, t)
    ref = [value(l, o[i], t[i]) for i in 1:length(o)]
    @test @inferred(value(l, o, t, AggMode.None())) == ref
    @test @inferred(value(l, o, t)) == ref
    @test value.(l, o, t) == ref
    @test @inferred(l(o, t)) == ref
    n = length(ref)
    s = sum(ref)
    @test @inferred(value(l, o, t, AggMode.Sum())) ≈ s
    @test @inferred(value(l, o, t, AggMode.Mean())) ≈ s / n
    ## Weighted Sum
    @test @inferred(value(l, o, t, AggMode.WeightedSum(ones(n)))) ≈ s
    @test @inferred(value(l, o, t, AggMode.WeightedSum(ones(n),normalize=true))) ≈ s / n
    ## Weighted Mean
    @test @inferred(value(l, o, t, AggMode.WeightedMean(ones(n)))) ≈ (s / n) / n
    @test @inferred(value(l, o, t, AggMode.WeightedMean(ones(n),normalize=false))) ≈ s / n
end

function test_vector_deriv(l, o, t)
    ref = [deriv(l, o[i], t[i]) for i in 1:length(o)]
    @test @inferred(deriv(l, o, t, AggMode.None())) == ref
    @test @inferred(deriv(l, o, t)) == ref
    @test deriv.(Ref(l), o, t) == ref
end

function test_vector_deriv2(l, o, t)
    ref = [deriv2(l, o[i], t[i]) for i in 1:length(o)]
    @test @inferred(deriv2(l, o, t, AggMode.None())) == ref
    @test @inferred(deriv2(l, o, t)) == ref
    @test deriv2.(Ref(l), o, t) == ref
end

@testset "Vectorized API" begin
    for T in (Float32, Float64)
        for O in (Float32, Float64)
            @testset "Margin-based $T -> $O" begin
                for (targets,outputs) in (
                        (rand(T[-1, 1], 4), (rand(O, 4) .- O(.5)) .* O(20)),
                    )
                    for loss in (LogitMarginLoss(),ModifiedHuberLoss(),
                                 L1HingeLoss(),SigmoidLoss())
                        @testset "$(loss): " begin
                            test_vector_value(loss, outputs, targets)
                            test_vector_deriv(loss, outputs, targets)
                            test_vector_deriv2(loss, outputs, targets)
                        end
                    end
                end
            end
            @testset "Distance-based $T -> $O" begin
                for (targets,outputs) in (
                        ((rand(T, 4) .- T(.5)) .* T(20), (rand(O, 4) .- O(.5)) .* O(20)),
                    )
                    for loss in (QuantileLoss(0.75),L2DistLoss(),
                                 EpsilonInsLoss(1))
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