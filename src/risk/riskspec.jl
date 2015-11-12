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

# ==========================================================================
# * generic predictor
# * generic penalty

@inline function value{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray,
    ŷ::AbstractMatrix = value(risk.predictor, X, w))
  res = meanvalue(risk.loss, y, ŷ)
  res += value(risk.penalty, w, size(X, 1))
  res
end

@inline function value!{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
    buffer::AbstractMatrix,
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray)
  value!(buffer, risk.predictor, X, w)
  res = meanvalue(risk.loss, y, buffer)
  res += value(risk.penalty, w, size(X, 1))
  res
end

@inline function grad{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
    risk::RiskModel{TPred, TLoss, TPen},
    X::AbstractMatrix,
    w::AbstractArray,
    y::AbstractArray,
    ŷ::AbstractArray = value(risk.predictor, X, w))
  dloss = deriv(risk.loss, y, ŷ)
  dpred = grad(risk.predictor, X, w)
  buffer = A_mul_Bt(dpred, dloss)
  broadcast!(*, buffer, buffer, 1/length(y))
  addgrad!(buffer, risk.penalty, w, size(X, 1))
end

@inline function grad!{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
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
  addgrad!(buffer, risk.penalty, w, size(X, 1))
end

# ==========================================================================
# * linear predictor
# * generic penalty

@inline function grad{INTERCEPT, TLoss<:Loss, TPen<:Penalty}(
        risk::RiskModel{LinearPredictor{INTERCEPT}, TLoss, TPen},
        X::AbstractMatrix,
        w::AbstractVector,
        y::AbstractArray,
        ŷ::AbstractArray = value(risk.predictor, X, w))
    buffer = zeros(length(w), 1)
    grad!(buffer, risk, X, w, y, ŷ)
end

@inline function grad!{T, INTERCEPT, TLoss<:Loss, TPen<:Penalty}(
        buffer::AbstractMatrix{T},
        risk::RiskModel{LinearPredictor{INTERCEPT}, TLoss, TPen},
        X::AbstractMatrix,
        w::AbstractArray,
        y::AbstractArray,
        ŷ::AbstractArray = value(risk.predictor, X, w))
    n = size(X, 2)
    k = size(X, 1)
    @_dimcheck length(y) == length(ŷ) == n && length(buffer) == length(w)
    @_dimcheck length(buffer) == (INTERCEPT ? k+1 : k)
    fill!(buffer, zero(T))
    @inbounds for i = 1:n
        dloss = deriv(risk.loss, y[i], ŷ[i])
        for j = 1:k
            tmp = dloss
            tmp *= X[j, i]
            tmp /= n
            buffer[j] += tmp
        end
        if INTERCEPT
            tmp = dloss
            tmp *= risk.predictor.bias
            tmp /= n
            buffer[k+1] += tmp
        end
    end
    addgrad!(buffer, risk.penalty, w, k)
end
