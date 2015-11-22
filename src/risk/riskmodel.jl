type RiskModel{TRisk<:EmpiricalRisk, TDetails, XT<:DenseMatrix, YT<:AbstractVector} <: RegressionModel
  params::TRisk
  details::TDetails
  Xtrain::XT
  Ytrain::YT
end

@inline nobs(fit::RiskModel) = length(fit.Ytrain)
@inline features(fit::RiskModel) = fit.Xtrain
@inline targets(fit::RiskModel) = fit.Ytrain
@inline model_response(fit::RiskModel) = fit.Ytrain

@inline details(fit::RiskModel) = fit.details
@inline isconverged(fit::RiskModel) = isconverged(details(fit))
@inline iterations(fit::RiskModel) = iterations(details(fit))
@inline params(fit::RiskModel) = fit.params
@inline minimum(fit::RiskModel) = minimum(details(fit))
@inline minimizer(fit::RiskModel) = minimizer(details(fit))
@inline intercept(fit::RiskModel) = typeof(predmodel(fit)) <: LinearPredictor{true}
#@inline predmodel(fit::RiskModel) = fit.predmodel
@inline coef(fit::RiskModel) = coef(details(fit))

@inline predict(fit::RiskModel) = predict(fit, features(fit))
#@inline classify{TRisk<:EmpiricalRiskClassifier}(fit::RiskModel{TRisk}) = classify(fit, features(fit))
#@inline accuracy{TRisk<:EmpiricalRiskClassifier}(fit::RiskModel{TRisk}) = accuracy(fit, features(fit), targets(fit))
