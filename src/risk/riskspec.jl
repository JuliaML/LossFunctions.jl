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

# ==========================================================================
# * generic predictor
# * no penalty

@inline function value{TPred<:Predictor, TLoss<:Loss, TPen<:NoPenalty}(
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray,
    ŷ::AbstractMatrix = value(risk.predictor, X, w))
  meanvalue(risk.loss, y, ŷ)
end

@inline function value!{TPred<:Predictor, TLoss<:Loss, TPen<:NoPenalty}(
    buffer::AbstractMatrix,
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray)
  value!(buffer, risk.predictor, X, w)
  meanvalue(risk.loss, y, buffer)
end

@inline function grad{TPred<:Predictor, TLoss<:Loss, TPen<:NoPenalty}(
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray,
    ŷ::AbstractArray = value(risk.predictor, X, w))
  dloss = deriv(risk.loss, y, ŷ)
  dpred = grad(risk.predictor, X, w)
  A_mul_Bt(dpred, dloss) .* (1/length(y))
end

@inline function grad!{TPred<:Predictor, TLoss<:Loss, TPen<:NoPenalty}(
    buffer::AbstractMatrix,
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray,
    ŷ::AbstractArray = value(risk.predictor, X, w))
  dloss = deriv(risk.loss, y, ŷ)
  dpred = grad(risk.predictor, X, w)
  A_mul_Bt!(buffer, dpred, dloss)
  broadcast!(*, buffer, buffer, 1/length(y))
end

# ==========================================================================
# * generic predictor
# * generic penalty

# ==========================================================================
# * linear predictor
# * no penalty

# ==========================================================================
# * linear predictor
# * generic penalty
