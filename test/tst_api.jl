function test_vector_value(l::MarginLoss, t, y)
    ref = [value(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    buf = fill!(similar(ref), 0)
    @test @inferred(value!(buf, l, t, y)) == ref
    @test buf == ref
    buf2 = fill!(similar(ref), 0)
    @test @inferred(value!(buf2, l, t, y, AggMode.None())) == ref
    @test buf2 == ref
    @test @inferred(value(l, t, y, AggMode.None())) == ref
    @test @inferred(value(l, t, y)) == ref
    @test value.(Ref(l), t, y) == ref
    # Sum
    s = sum(ref)
    @test @inferred(value(l, t, y, AggMode.Sum())) ≈ s
    # Mean
    m = mean(ref)
    @test @inferred(value(l, t, y, AggMode.Mean())) ≈ m
    # Obs specific
    n = size(t, 1)
    k = size(t, ndims(t))
    ## Weighted Mean
    @test_throws DimensionMismatch value(l, t, y, AggMode.WeightedMean(ones(n-1)), ObsDim.First())
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(n)), ObsDim.First())) ≈ m
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(k)), ObsDim.Last())) ≈ m
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(k)))) ≈ m
    ## Weighted Sum
    @test_throws DimensionMismatch value(l, t, y, AggMode.WeightedSum(ones(n-1)), ObsDim.First())
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(n)), ObsDim.First())) ≈ s
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(k)), ObsDim.Last())) ≈ s
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(k)))) ≈ s
    if typeof(t) <: AbstractVector
        @test_throws ArgumentError value(l, t, y, AggMode.Sum(), ObsDim.First())
        @test_throws ArgumentError value(l, t, y, AggMode.Mean(), ObsDim.First())
    else
        # Sum per obs
        sv = vec(sum(ref, dims=1:(ndims(ref)-1)))
        @test @inferred(value(l, t, y, AggMode.Sum(), ObsDim.Last())) ≈ sv
        buffer1 = fill!(similar(sv), 0)
        @test @inferred(value!(buffer1, l, t, y, AggMode.Sum(), ObsDim.Last())) ≈ sv
        @test buffer1 ≈ sv
        buffer2 = fill!(similar(sv), 0)
        # Weighted sum compare
        @test @inferred(value(l, t, y, AggMode.WeightedSum(1:k), ObsDim.Last())) ≈ sum(sv .* (1:k))
        @test round(@inferred(value(l, t, y, AggMode.WeightedSum(1:k,normalize=true), ObsDim.Last())), digits=3) ≈
            round(sum(sv .* ((1:k)/(sum(1:k)))), digits=3)
        # Mean per obs
        mv = vec(mean(ref, dims=1:(ndims(ref)-1)))
        @test @inferred(value(l, t, y, AggMode.Mean(), ObsDim.Last())) ≈ mv
        buffer3 = fill!(similar(mv), 0)
        @test @inferred(value!(buffer3, l, t, y, AggMode.Mean(), ObsDim.Last())) ≈ mv
        @test buffer3 ≈ mv
        # Weighted mean compare
        @test @inferred(value(l, t, y, AggMode.WeightedMean(1:k,normalize=false), ObsDim.Last())) ≈ sum(mv .* (1:k))
    end
end

function test_vector_value(l::DistanceLoss, t, y)
    ref = [value(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    buf = fill!(similar(ref), 0)
    @test @inferred(value!(buf, l, t, y)) == ref
    @test buf == ref
    buf2 = fill!(similar(ref), 0)
    @test @inferred(value!(buf2, l, t, y, AggMode.None())) == ref
    @test buf2 == ref
    @test @inferred(value(l, t, y, AggMode.None())) == ref
    @test @inferred(value(l, t, y)) == ref
    @test value.(Ref(l), t, y) == ref
    # Sum
    s = sum(ref)
    @test @inferred(value(l, t, y, AggMode.Sum())) ≈ s
    # Mean
    m = mean(ref)
    @test round(@inferred(value(l, t, y, AggMode.Mean())), digits=5) ≈ round(m, digits=5)
    # Obs specific
    n = size(t, 1)
    k = size(t, ndims(t))
    ## Weighted Mean
    @test_throws DimensionMismatch value(l, t, y, AggMode.WeightedMean(ones(n-1)), ObsDim.First())
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(n)), ObsDim.First())) ≈ m
    @test @inferred(value(l, t, y, AggMode.WeightedMean(ones(k)), ObsDim.Last())) ≈ m
    ## Weighted Sum
    @test_throws DimensionMismatch value(l, t, y, AggMode.WeightedSum(ones(n-1)), ObsDim.First())
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(n),normalize=false), ObsDim.First())) ≈ s
    @test @inferred(value(l, t, y, AggMode.WeightedSum(ones(k),normalize=false), ObsDim.Last())) ≈ s
    if typeof(t) <: AbstractVector
        @test_throws ArgumentError value(l, t, y, AggMode.Sum(), ObsDim.First())
        @test_throws ArgumentError value(l, t, y, AggMode.Mean(), ObsDim.First())
    else
        # Sum per obs
        sv = vec(sum(ref, dims=1:(ndims(ref)-1)))
        @test @inferred(value(l, t, y, AggMode.Sum(), ObsDim.Last())) ≈ sv
        buffer1 = fill!(similar(sv), 0)
        @test @inferred(value!(buffer1, l, t, y, AggMode.Sum(), ObsDim.Last())) ≈ sv
        @test buffer1 ≈ sv
        # Weighted sum compare
        @test @inferred(value(l, t, y, AggMode.WeightedSum(1:k,normalize=false), ObsDim.Last())) ≈ sum(sv .* (1:k))
        # Mean per obs
        mv = vec(mean(ref, dims=1:(ndims(ref)-1)))
        @test @inferred(value(l, t, y, AggMode.Mean(), ObsDim.Last())) ≈ mv
        buffer3 = fill!(copy(mv), 0)
        @test @inferred(value!(buffer3, l, t, y, AggMode.Mean(), ObsDim.Last())) ≈ mv
        @test buffer3 ≈ mv
        # Weighted mean compare
        @test @inferred(value(l, t, y, AggMode.WeightedMean(1:k,normalize=false), ObsDim.Last())) ≈ sum(mv .* (1:k))
    end
end

function test_vector_deriv(l::MarginLoss, t, y)
    ref = [LossFunctions.deriv(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(LossFunctions.deriv(l, t, y, AggMode.None())) == ref
    @test @inferred(LossFunctions.deriv(l, t, y)) == ref
    @test LossFunctions.deriv.(Ref(l), t, y) == ref
end

function test_vector_deriv(l::DistanceLoss, t, y)
    ref = [LossFunctions.deriv(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(LossFunctions.deriv(l, t, y, AggMode.None())) == ref
    @test @inferred(LossFunctions.deriv(l, t, y)) == ref
    @test LossFunctions.deriv.(Ref(l), t, y) == ref
end

function test_vector_deriv2(l::MarginLoss, t, y)
    ref = [LossFunctions.deriv2(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(LossFunctions.deriv2(l, t, y, AggMode.None())) == ref
    @test @inferred(LossFunctions.deriv2(l, t, y)) == ref
    @test LossFunctions.deriv2.(Ref(l), t, y) == ref
end

function test_vector_deriv2(l::DistanceLoss, t, y)
    ref = [LossFunctions.deriv2(l,t[i],y[i]) for i in CartesianIndices(size(y))]
    @test @inferred(LossFunctions.deriv2(l, t, y, AggMode.None())) == ref
    @test @inferred(LossFunctions.deriv2(l, t, y)) == ref
    @test LossFunctions.deriv2.(Ref(l), t, y) == ref
end

@testset "Vectorized API" begin
    for T in (Float32, Float64)
        for O in (Float32, Float64)
            @testset "Margin-based $T -> $O" begin
                for (targets,outputs) in (
                        (rand(T[-1, 1], 4), (rand(O, 4) .- O(.5)) .* O(20)),
                        (rand(T[-1, 1], 4, 5), (rand(O, 4, 5) .- O(.5)) .* O(20)),
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
                        ((rand(T, 4, 5) .- T(.5)) .* T(20), (rand(O, 4, 5) .- O(.5)) .* O(20)),
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

@testset "Broadcasting higher-order arrays" begin
    for f in (value,deriv,deriv2)
        @testset "$f" begin

            m,n,k = 100,99,98
            loss = L2DistLoss()

            targ1 = randn(m)
            targ2 = repeat(targ1,outer=(1,n))
            targ3 = repeat(targ1,outer=(1,n,k))

            out1 = randn(m,n)
            out2 = randn(m,n,k)

            for avg in (AggMode.None(),AggMode.Mean(),AggMode.Sum())
                @test f(loss,targ1,out1,avg) ≈ f(loss,targ2,out1,avg)
                @test f(loss,targ1,out2,avg) ≈ f(loss,targ2,out2,avg)
                @test f(loss,targ1,out2,avg) ≈ f(loss,targ3,out2,avg)
                @test f(loss,targ2,out2,avg) ≈ f(loss,targ3,out2,avg)
                @test f(loss,out1,targ1,avg) ≈ f(loss,out1,targ2,avg)
                @test f(loss,out2,targ1,avg) ≈ f(loss,out2,targ2,avg)
                @test f(loss,out2,targ1,avg) ≈ f(loss,out2,targ3,avg)
                @test f(loss,out2,targ2,avg) ≈ f(loss,out2,targ3,avg)
            end
        end
    end
end
