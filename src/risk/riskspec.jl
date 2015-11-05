abstract EmpiricalRisk{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}

immutable RiskModel{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty} <: EmpiricalRisk{TPred, TLoss, TPen}
  predictor::TPred
  loss::TLoss
  penalty::TPen
end

function RiskModel{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
    predictor::TPred = LinearPredictor(0),
    loss::TLoss = L2DistLoss(),
    penalty::TPen = NoPenalty())
  RiskModel{TPred, TLoss, TPen}(predictor, loss, penalty)
end

function value{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractVector)
  
end
