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

            # can't broadcast in this direction (yet)
            @test_throws Exception f(loss,targ3,out1)
        end
    end
end
