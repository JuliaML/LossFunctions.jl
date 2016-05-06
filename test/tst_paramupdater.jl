
for T in (SGDUpdater,
          AdagradUpdater,
          AdadeltaUpdater,
          AdamUpdater,
          AdaMaxUpdater)

    @testset "$(string(T))" begin
        updater = T()
        state = ParameterUpdaterState(T)
        ploss = L2ParameterLoss(1e-5)

        @test typeof(state) == get_state_type(T)
        @test typeof(ParameterUpdaterState(updater, 1, 1)) == Matrix{typeof(state)}

        # make sure the function doesn't error, and is type stable
        for T in (Int, Int32, Float64, Float32)
            @test typeof(param_change!(state, updater, one(T))) == Float64
            @test typeof(param_change!(state, updater, ploss, one(T), 0.01)) == Float64
        end
    end
end
