function test_vector_value(l, t, y)
    ref = [value(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(value(l, t, y, AggMode.None())) == ref
    @test @inferred(value(l, t, y)) == ref
    @test value.(l, t, y) == ref
    @test @inferred(l(t, y)) == ref
    n = length(ref)
    s = sum(ref)
    @test @inferred(value(l, t, y, AggMode.Sum())) ≈ s
    @test @inferred(value(l, t, y, AggMode.Mean())) ≈ s / n
    ## Weighted Sum
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(n)))) ≈ s
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(n),normalize=true))) ≈ s / n
    ## Weighted Mean
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(n)))) ≈ (s / n) / n
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(n),normalize=false))) ≈ s / n
end

function test_vector_deriv(l, t, y)
    ref = [deriv(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(deriv(l, t, y, AggMode.None())) == ref
    @test @inferred(deriv(l, t, y)) == ref
    @test deriv.(Ref(l), t, y) == ref
end

function test_vector_deriv2(l, t, y)
    ref = [deriv2(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(deriv2(l, t, y, AggMode.None())) == ref
    @test @inferred(deriv2(l, t, y)) == ref
    @test deriv2.(Ref(l), t, y) == ref
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
                            test_vector_value(loss, targets, outputs)
                            test_vector_deriv(loss, targets, outputs)
                            test_vector_deriv2(loss, targets, outputs)
                        end
                    end
                end
            end
            println("<HEARTBEAT>")
            @testset "Distance-based $T -> $O" begin
                for (targets,outputs) in (
                        ((rand(T, 4) .- T(.5)) .* T(20), (rand(O, 4) .- O(.5)) .* O(20)),
                    )
                    for loss in (QuantileLoss(0.75),L2DistLoss(),
                                 EpsilonInsLoss(1))
                        @testset "$(loss): " begin
                            test_vector_value(loss, targets, outputs)
                            test_vector_deriv(loss, targets, outputs)
                            test_vector_deriv2(loss, targets, outputs)
                        end
                    end
                end
            end
            println("<HEARTBEAT>")
        end
    end
end