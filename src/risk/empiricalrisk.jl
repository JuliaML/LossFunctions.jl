# using LearnBase.LossFunctions
using LearnBase.ParameterLosses

immutable EmpiricalRisk{TPred<:Predictor, TLoss<:ModelLoss, TPen<:ParameterLoss, PENALIZEBIAS}
    predictor::TPred
    loss::TLoss
    penalty::TPen
end

function EmpiricalRisk{TPred<:Predictor, TLoss<:ModelLoss, TPen<:ParameterLoss}(
        predictor::TPred = LinearPredictor(0),
        loss::TLoss = L2DistLoss(),
        penalty::TPen = NoParameterLoss(),
        penalize_bias::Bool = false)
    EmpiricalRisk{TPred, TLoss, TPen, penalize_bias}(predictor, loss, penalty)
end

intercept(risk::EmpiricalRisk) = intercept(risk.predictor)

# ==========================================================================

typealias EmpiricalRiskClassifier{TPred<:Predictor, TLoss<:MarginLoss, TPen<:ParameterLoss} EmpiricalRisk{TPred, TLoss, TPen}
typealias EmpiricalRiskRegressor{TPred<:Predictor, TLoss<:DistanceLoss, TPen<:ParameterLoss} EmpiricalRisk{TPred, TLoss, TPen}

# ==========================================================================
# * generic predictor
# * no penalty

function value{TPred<:Predictor, TLoss<:ModelLoss, TPen<:NoParameterLoss}(
        risk::EmpiricalRisk{TPred, TLoss, TPen},
        X::AbstractArray,
        w::AbstractArray,
        y,
        ŷ = value(risk.predictor, X, w))
    meanvalue(risk.loss, y, ŷ)
end

function value!{TPred<:Predictor, TLoss<:ModelLoss, TPen<:NoParameterLoss}(
        buffer::AbstractMatrix,
        risk::EmpiricalRisk{TPred, TLoss, TPen},
        X::AbstractArray,
        w::AbstractArray,
        y)
    value!(buffer, risk.predictor, X, w)
    meanvalue(risk.loss, y, buffer)
end

# ==========================================================================
# * generic predictor
# * generic penalty

function value{TPred<:Predictor, TLoss<:ModelLoss, TPen<:ParameterLoss, PENALIZEBIAS}(
        risk::EmpiricalRisk{TPred, TLoss, TPen, PENALIZEBIAS},
        X::AbstractArray,
        w::AbstractArray,
        y,
        ŷ = value(risk.predictor, X, w))
    res = meanvalue(risk.loss, y, ŷ)
    if PENALIZEBIAS
        res += value(risk.penalty, w)
    else
        res += value(risk.penalty, w, size(X, 1))
    end
    res
end

function value!{TPred<:Predictor, TLoss<:ModelLoss, TPen<:ParameterLoss, PENALIZEBIAS}(
        buffer::AbstractMatrix,
        risk::EmpiricalRisk{TPred, TLoss, TPen, PENALIZEBIAS},
        X::AbstractArray,
        w::AbstractArray,
        y)
    value!(buffer, risk.predictor, X, w)
    res = meanvalue(risk.loss, y, buffer)
    if PENALIZEBIAS
        res += value(risk.penalty, w)
    else
        res += value(risk.penalty, w, size(X, 1))
    end
    res
end

function grad{TPred<:Predictor, TLoss<:ModelLoss, TPen<:ParameterLoss, PENALIZEBIAS}(
        risk::EmpiricalRisk{TPred, TLoss, TPen, PENALIZEBIAS},
        X::AbstractArray,
        w::AbstractArray,
        y,
        ŷ = value(risk.predictor, X, w))
    dloss = deriv(risk.loss, y, ŷ)
    dpred = grad(risk.predictor, X, w)
    buffer = A_mul_Bt(dpred, dloss)
    broadcast!(*, buffer, buffer, 1/length(y))
    if PENALIZEBIAS
        addgrad!(buffer, risk.penalty, w)
    else
        addgrad!(buffer, risk.penalty, w, size(X, 1))
    end
end

function grad!{TPred<:Predictor, TLoss<:ModelLoss, TPen<:ParameterLoss, PENALIZEBIAS}(
        buffer::AbstractMatrix,
        risk::EmpiricalRisk{TPred, TLoss, TPen, PENALIZEBIAS},
        X::AbstractArray,
        w::AbstractArray,
        y,
        ŷ = value(risk.predictor, X, w))
    dloss = deriv(risk.loss, y, ŷ)
    dpred = grad(risk.predictor, X, w)
    A_mul_Bt!(buffer, dpred, dloss)
    broadcast!(*, buffer, buffer, 1/length(y))
    if PENALIZEBIAS
        addgrad!(buffer, risk.penalty, w)
    else
        addgrad!(buffer, risk.penalty, w, size(X, 1))
    end
end

# ==========================================================================
# * linear predictor
# * generic penalty

function grad{INTERCEPT, TLoss<:ModelLoss, TPen<:ParameterLoss}(
        risk::EmpiricalRisk{LinearPredictor{INTERCEPT}, TLoss, TPen},
        X::AbstractArray,
        w::AbstractVector,
        y,
        ŷ = value(risk.predictor, X, w))
    buffer = zeros(length(w), 1)
    grad!(buffer, risk, X, w, y, ŷ)
end

function grad!{T, INTERCEPT, TLoss<:ModelLoss, TPen<:ParameterLoss, PENALIZEBIAS}(
        buffer::AbstractMatrix{T},
        risk::EmpiricalRisk{LinearPredictor{INTERCEPT}, TLoss, TPen, PENALIZEBIAS},
        X::AbstractArray,
        w::AbstractArray,
        y,
        ŷ = value(risk.predictor, X, w))
    n = size(X, 2)
    k = size(X, 1)
    @_dimcheck length(y) == length(ŷ) == n && length(buffer) == length(w)
    @_dimcheck length(buffer) == (INTERCEPT ? k+1 : k)
    fill!(buffer, zero(T))
    @inbounds for i = 1:n
        dloss = deriv(risk.loss, y[i], ŷ[i]) / n
        @simd for j = 1:k
            buffer[j] += dloss * X[j, i]
        end
        if INTERCEPT
            buffer[k+1] += dloss * risk.predictor.bias
        end
    end
    if PENALIZEBIAS
        addgrad!(buffer, risk.penalty, w)
    else
        addgrad!(buffer, risk.penalty, w, k)
    end
end
