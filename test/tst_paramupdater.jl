
updater = SGDUpdater()
@test typeof(updater.ploss) == NoParameterLoss
@test typeof(ParameterUpdaterState(updater)) == SGDState
@test typeof(ParameterUpdaterState(updater, 1, 1)) == Matrix{SGDState}
