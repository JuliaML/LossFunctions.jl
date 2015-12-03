using LearnBase.LossFunctions
using LearnBase.Penalties

immutable EmpiricalRisk{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, PENALIZEBIAS}
    predictor::TPred
    loss::TLoss
    penalty::TPen
end

function EmpiricalRisk{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
        predictor::TPred = LinearPredictor(0),
        loss::TLoss = L2DistLoss(),
        penalty::TPen = NoPenalty(),
        penalize_bias::Bool = false)
    EmpiricalRisk{TPred, TLoss, TPen, penalize_bias}(predictor, loss, penalty)
end

intercept(risk::EmpiricalRisk) = intercept(risk.predictor)

# ==========================================================================

typealias EmpiricalRiskClassifier{TPred<:Predictor, TLoss<:MarginBasedLoss, TPen<:Penalty} EmpiricalRisk{TPred, TLoss, TPen}
typealias EmpiricalRiskRegressor{TPred<:Predictor, TLoss<:DistanceBasedLoss, TPen<:Penalty} EmpiricalRisk{TPred, TLoss, TPen}

# ==========================================================================
# * generic predictor
# * no penalty

@inline function value{TPred<:Predictor, TLoss<:Loss, TPen<:NoPenalty}(
        risk::EmpiricalRisk{TPred, TLoss, TPen},
        X::AbstractArray,
        w::AbstractArray,
        y,
        ŷ = value(risk.predictor, X, w))
    meanvalue(risk.loss, y, ŷ)
end

@inline function value!{TPred<:Predictor, TLoss<:Loss, TPen<:NoPenalty}(
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

@inline function value{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, PENALIZEBIAS}(
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

@inline function value!{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, PENALIZEBIAS}(
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

@inline function grad{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, PENALIZEBIAS}(
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

@inline function grad!{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, PENALIZEBIAS}(
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

@inline function grad{INTERCEPT, TLoss<:Loss, TPen<:Penalty}(
        risk::EmpiricalRisk{LinearPredictor{INTERCEPT}, TLoss, TPen},
        X::AbstractArray,
        w::AbstractVector,
        y,
        ŷ = value(risk.predictor, X, w))
    buffer = zeros(length(w), 1)
    grad!(buffer, risk, X, w, y, ŷ)
end

@inline function grad!{T, INTERCEPT, TLoss<:Loss, TPen<:Penalty, PENALIZEBIAS}(
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
        dloss = deriv(risk.loss, y[i], ŷ[i])
        @simd for j = 1:k
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
    if PENALIZEBIAS
        addgrad!(buffer, risk.penalty, w)
    else
        addgrad!(buffer, risk.penalty, w, k)
    end
end
