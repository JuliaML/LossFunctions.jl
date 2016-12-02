function test_vector_value(l::MarginLoss, t, y)
    @testset "$(l): " begin
        ref = [ LossFunctions.value(l,t[i],y[i]) for i in 1:length(y) ]
        @test LossFunctions.value.(l, t, y) == ref
        @test LossFunctions.value.(l, t .* y) == ref
        @test l.(t, y) == ref
        @test l.(t .* y) == ref
    end
end

function test_vector_value(l::DistanceLoss, t, y)
    @testset "$(l): " begin
        ref = [ LossFunctions.value(l,t[i],y[i]) for i in 1:length(y) ]
        @test LossFunctions.value.(l, t, y) == ref
        @test LossFunctions.value.(l, y - t) == ref
        @test l.(t, y) == ref
        @test l.(y - t) == ref
    end
end

function test_vector_deriv(l::MarginLoss, t, y)
    @testset "$(l): " begin
        ref = [ LossFunctions.deriv(l,t[i],y[i]) for i in 1:length(y) ]
        @test LossFunctions.deriv.(l, t, y) == ref
        @test t .* LossFunctions.deriv.(l, t .* y) == ref
        @test l'.(t, y) == ref
        @test t .* l'.(t .* y) == ref
    end
end

function test_vector_deriv(l::DistanceLoss, t, y)
    @testset "$(l): " begin
        ref = [ LossFunctions.deriv(l,t[i],y[i]) for i in 1:length(y) ]
        @test LossFunctions.deriv.(l, t, y) == ref
        @test LossFunctions.deriv.(l, y - t) == ref
        @test l'.(t, y) == ref
        @test l'.(y - t) == ref
    end
end

function test_vector_deriv2(l::MarginLoss, t, y)
    @testset "$(l): " begin
        ref = [ LossFunctions.deriv2(l,t[i],y[i]) for i in 1:length(y) ]
        @test LossFunctions.deriv2.(l, t, y) == ref
        @test LossFunctions.deriv2.(l, t .* y) == ref
        @test l''.(t, y) == ref
        @test l''.(t .* y) == ref
    end
end

function test_vector_deriv2(l::DistanceLoss, t, y)
    @testset "$(l): " begin
        ref = [ LossFunctions.deriv2(l,t[i],y[i]) for i in 1:length(y) ]
        @test LossFunctions.deriv2.(l, t, y) == ref
        @test LossFunctions.deriv2.(l, y - t) == ref
        @test l''.(t, y) == ref
        @test l''.(y - t) == ref
    end
end

@testset "Vectorized API" begin
    targets = rand([-1,1], 10)
    outputs = (rand(10)-.5) * 20

    for loss in margin_losses
        test_vector_value(loss, targets, outputs)
        test_vector_deriv(loss, targets, outputs)
        test_vector_deriv2(loss, targets, outputs)
    end

    targets = (rand(10)-.5) * 20
    outputs = (rand(10)-.5) * 20
    for loss in distance_losses
        test_vector_value(loss, targets, outputs)
        test_vector_deriv(loss, targets, outputs)
        test_vector_deriv2(loss, targets, outputs)
    end
end

@testset "Broadcasting higher-order arrays" begin
    for f in (LossFunctions.value,deriv,sumvalue,sumderiv,meanvalue,meanderiv)
        @testset "$f" begin

            m,n,k = 100,99,98
            loss = L2DistLoss()

            targ1 = randn(m)
            targ2 = repeat(targ1,outer=(1,n))
            targ3 = repeat(targ1,outer=(1,n,k))

            out1 = randn(m,n)
            out2 = randn(m,n,k)

            @test isapprox(f(loss,targ1,out1), f(loss,targ2,out1))
            @test isapprox(f(loss,targ1,out2), f(loss,targ2,out2))
            @test isapprox(f(loss,targ1,out2), f(loss,targ3,out2))
            @test isapprox(f(loss,targ2,out2), f(loss,targ3,out2))
        end
    end
end

