function test_value_typestable(l::SupervisedLoss)
    @testset "$(l): " begin
        for y in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-.5), Float32(.5))
            for t in (-2, 2, Int32(-1), Int32(1), -.5, .5, Float32(-1), Float32(1))
                # get expected return type
                T = promote_type(typeof(y), typeof(t))

                # test basic loss
                val = value(l, y, t)
                @test typeof(val) <: T

                # test scaled version of loss
                @test typeof(value(T(2)*l, y, t)) <: T
            end
        end
    end
end

function test_value_float32_preserving(l::SupervisedLoss)
    @testset "$(l): " begin
        for y in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-.5), Float32(.5))
            for t in (-2, 2, Int32(-1), Int32(1), -.5, .5, Float32(-1), Float32(1))
                val = value(l, y, t)
                T = promote_type(typeof(y),typeof(t))
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
        for y in (-1, 1, Int32(-1), Int32(1), -1.5, 1.5, Float32(-.5), Float32(.5))
            for t in (-2, 2, Int32(-1), Int32(1), -.5, .5, Float32(-1), Float32(1))
                val = value(l, y, t)
                @test (typeof(val) <: Float64)
            end
        end
    end
end

function test_value(l::SupervisedLoss, f::Function, y_vec, t_vec)
    @testset "$(l): " begin
        for y in y_vec, t in t_vec
            @test abs(value(l, y, t) - f(y, t)) < 1e-10
            # TODO: consider this test?
            # @test abs(value_deriv(l,y,t)[1] - f(y, t)) < 1e-10
        end
    end
end

function test_deriv(l::MarginLoss, t_vec)
    @testset "$(l): " begin
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
                @test_approx_eq val value_fun(l)(y, t)
                @test_approx_eq val value_fun(l)(y*t)
                @test_approx_eq d_comp d_comp2
                @test_approx_eq d_comp d_comp3
                @test_approx_eq d_comp y*d_comp4
                @test_approx_eq d_comp y*deriv(l, y*t)
                @test_approx_eq d_comp deriv_fun(l)(y, t)
                @test_approx_eq d_comp y*deriv_fun(l)(y*t)
            else
                # y*t == 1 ? print(".") : print("(no $(y)*$(t)) ")
                #print(".")
            end
        end
    end
end

function test_deriv(l::DistanceLoss, t_vec)
    @testset "$(l): " begin
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
                @test_approx_eq val value_fun(l)(y, t)
                @test_approx_eq val value_fun(l)(t-y)
                @test_approx_eq d_comp d_comp2
                @test_approx_eq d_comp d_comp3
                @test_approx_eq d_comp d_comp4
                @test_approx_eq d_comp deriv(l, t-y)
                @test_approx_eq d_comp deriv_fun(l)(y, t)
                @test_approx_eq d_comp deriv_fun(l)(t-y)
            else
                # y-t == 0 ? print(".") : print("$(y-t) ")
                #print(".")
            end
        end
    end
end

function test_deriv2(l::MarginLoss, t_vec)
    @testset "$(l): " begin
        for y in [-1., 1], t in t_vec
            if istwicedifferentiable(l, y*t)
                d2_dual = epsilon(deriv(l, dual(y, 0), dual(t, 1)))
                d2_comp = deriv2(l, y, t)
                @test abs(d2_dual - d2_comp) < 1e-10
                @test_approx_eq d2_comp deriv2(l, y, t)
                @test_approx_eq d2_comp deriv2(l, y*t)
                @test_approx_eq d2_comp deriv2_fun(l)(y, t)
                @test_approx_eq d2_comp deriv2_fun(l)(y*t)
            else
                # y*t == 1 ? print(".") : print("(no $(y)*$(t)) ")
                #print(".")
            end
        end
    end
end

function test_deriv2(l::DistanceLoss, t_vec)
    @testset "$(l): " begin
        for y in -20:.2:20, t in t_vec
            if istwicedifferentiable(l, t-y)
                d2_dual = epsilon(deriv(l, dual(t-y, 1)))
                d2_comp = deriv2(l, y, t)
                @test abs(d2_dual - d2_comp) < 1e-10
                @test_approx_eq d2_comp deriv2(l, y, t)
                @test_approx_eq d2_comp deriv2(l, t-y)
                @test_approx_eq d2_comp deriv2_fun(l)(y, t)
                @test_approx_eq d2_comp deriv2_fun(l)(t-y)
            else
                # y-t == 0 ? print(".") : print("$(y-t) ")
                #print(".")
            end
        end
    end
end

function test_scaledloss(l::Loss, t_vec, y_vec)
    @testset "Scaling for $(l): " begin
        for λ = (2.0, 2)
            sl = ScaledLoss(l,λ)
            @test sl == λ * l
            for t in t_vec
                for y in y_vec
                    @test value(ScaledLoss(l,λ),t,y) == λ*value(l,t,y)
                    @test deriv(ScaledLoss(l,λ),t,y) == λ*deriv(l,t,y)
                    @test deriv2(ScaledLoss(l,λ),t,y) == λ*deriv2(l,t,y)
                end
            end
        end
    end
end

function test_scaledloss(l::Loss, n_vec)
    @testset "Scaling for $(l): " begin
        for λ = (2.0, 2)
            sl = ScaledLoss(l,λ)
            @test sl == λ * l
            for n in n_vec
                @test value(ScaledLoss(l,λ),n) == λ*value(l,n)
                @test deriv(ScaledLoss(l,λ),n) == λ*deriv(l,n)
                @test deriv2(ScaledLoss(l,λ),n) == λ*deriv2(l,n)
            end
        end
    end
end

# ====================================================================

@testset "Test typestable supervised loss for type stability" begin
    for loss in [L1HingeLoss(), L2HingeLoss(), ModifiedHuberLoss(), PerceptronLoss(),
                LPDistLoss(1), LPDistLoss(2), LPDistLoss(3)]
        test_value_typestable(loss)
        # TODO: add ZeroOneLoss after scaling works...
    end
end

@testset "Test float-forcing supervised loss for type stability" begin
    # Losses that should always return Float64
    for loss in [SmoothedL1HingeLoss(0.5), SmoothedL1HingeLoss(1), L1EpsilonInsLoss(0.5),
                 L1EpsilonInsLoss(1), L2EpsilonInsLoss(0.5), L2EpsilonInsLoss(1), PeriodicLoss(1)]
        test_value_float64_forcing(loss)
        test_value_float64_forcing(2.0 * loss)
    end
    test_value_float64_forcing(2.0 * LogitDistLoss())
    test_value_float64_forcing(2.0 * LogitMarginLoss())
    
    # Losses that should return an AbstractFloat, preserving type if possible
    for loss in [PeriodicLoss(Float32(1)), PeriodicLoss(Float32(0.5)),
                 LogitDistLoss(), LogitMarginLoss(),
                 L1EpsilonInsLoss(Float32(1)), L1EpsilonInsLoss(Float32(0.5)),
                 L2EpsilonInsLoss(Float32(1)), L2EpsilonInsLoss(Float32(0.5))]
        test_value_float32_preserving(loss)
        test_value_float32_preserving(Float32(2) * loss)
    end
end

@testset "Test margin-based loss against reference function" begin
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
end

@testset "Test distance-based loss against reference function" begin
    yr,tr = linspace(-20,20,10),linspace(-30,30,10)

    _l1distloss(y, t) = abs(t - y)
    test_value(L1DistLoss(), _l1distloss, yr, tr)

    _l2distloss(y, t) = (t - y)^2
    test_value(L2DistLoss(), _l2distloss, yr, tr)

    _lp15distloss(y, t) = abs(t - y)^(1.5)
    test_value(LPDistLoss(1.5), _lp15distloss, yr, tr)

    function _periodicloss(c)
        _value(y, t) = 1 - cos((y-t)*2π/c)
        _value
    end
    test_value(PeriodicLoss(0.5), _periodicloss(0.5), yr, tr)
    test_value(PeriodicLoss(1), _periodicloss(1), yr, tr)
    test_value(PeriodicLoss(1.5), _periodicloss(1.5), yr, tr)

    function _l1epsinsloss(ɛ)
        _value(y, t) = max(0, abs(t - y) - ɛ)
        _value
    end
    test_value(EpsilonInsLoss(0.5), _l1epsinsloss(0.5), yr, tr)
    test_value(EpsilonInsLoss(1), _l1epsinsloss(1), yr, tr)
    test_value(EpsilonInsLoss(1.5), _l1epsinsloss(1.5), yr, tr)

    function _l2epsinsloss(ɛ)
        _value(y, t) = max(0, abs(t - y) - ɛ)^2
        _value
    end
    test_value(L2EpsilonInsLoss(0.5), _l2epsinsloss(0.5), yr, tr)
    test_value(L2EpsilonInsLoss(1), _l2epsinsloss(1), yr, tr)
    test_value(L2EpsilonInsLoss(1.5), _l2epsinsloss(1.5), yr, tr)

    _logitdistloss(y, t) = -log((4*exp(t-y))/(1+exp(t-y))^2)
    test_value(LogitDistLoss(), _logitdistloss, yr, tr)
end

@testset "Test other loss against reference function" begin
    _crossentropyloss(y, t) = -y*log(t) - (1-y)*log(1-t)
    test_value(CrossentropyLoss(), _crossentropyloss, 0:0.01:1, 0.01:0.01:0.99)

    _zerooneloss(y, t) = sign(y*t) < 0 ? 1 : 0
    test_value(ZeroOneLoss(), _zerooneloss, [-1.,1], -10:0.1:10)
end

margin_losses = [LogitMarginLoss(), L1HingeLoss(), L2HingeLoss(), PerceptronLoss(),
                 SmoothedL1HingeLoss(.5), SmoothedL1HingeLoss(1), SmoothedL1HingeLoss(2),
                 ModifiedHuberLoss()]

@testset "Test first derivatives of margin-based losses" begin
    for loss in margin_losses
        test_deriv(loss, -10:0.1:10)
    end
end

@testset "Test second derivatives of margin-based losses" begin
    for loss in margin_losses
        test_deriv2(loss, -10:0.1:10)
    end
end

@testset "Test margin-based scaled loss" begin
    for loss in margin_losses
        test_scaledloss(loss, [-1.,1], -10:0.1:10)
        test_scaledloss(loss, -10:0.1:10)
    end
end

distance_losses = [L2DistLoss(), LPDistLoss(2.0), L1DistLoss(), LPDistLoss(1.0),
                   LPDistLoss(0.5), LPDistLoss(1.5), LPDistLoss(3),
                   LogitDistLoss(), L1EpsilonInsLoss(0.5), EpsilonInsLoss(1.5),
                   L2EpsilonInsLoss(0.5), L2EpsilonInsLoss(1.5), PeriodicLoss(1)]

@testset "Test first derivatives of distance-based losses" begin
    for loss in distance_losses
        test_deriv(loss, -30:0.5:30)
    end
end

@testset "Test second derivatives of distance-based losses" begin
    for loss in distance_losses
        test_deriv2(loss, -30:0.5:30)
    end
end

@testset "Test distance-based scaled loss" begin
    for loss in distance_losses
        test_scaledloss(loss, -20:.2:20, -30:0.5:30)
        test_scaledloss(loss, -30:0.5:30)
    end
end

