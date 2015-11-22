
type RiskFunctional{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, TX<:AbstractArray, TY<:AbstractArray} <: EmpiricalRisk{TPred, TLoss, TPen}
    model::RiskModel{TPred, TLoss, TPen}
    X::TX
    Y::TY
end

function RiskFunctional{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
        model::RiskModel{TPred, TLoss, TPen},
        X::AbstractArray,
        Y::AbstractArray)
    RiskFunctional{TPred, TLoss, TPen, typeof(X), typeof(Y)}(model, X, Y)
end

function value(risk::RiskFunctional, w::AbstractArray, ŷ::AbstractMatrix = value(risk.model.predictor, risk.X, w))
    value(risk.model, risk.X, w, risk.Y, ŷ)
end

function value!(buffer::AbstractMatrix, risk::RiskFunctional, w::AbstractArray)
    value!(buffer, risk.model, risk.X, w, risk.Y)
end

function grad(risk::RiskFunctional, w::AbstractArray, ŷ::AbstractMatrix = value(risk.model.predictor, risk.X, w))
    grad(risk.model, risk.X, w, risk.Y, ŷ)
end

function grad!(buffer::AbstractMatrix, risk::RiskFunctional, w::AbstractArray, ŷ::AbstractMatrix = value(risk.model.predictor, risk.X, w))
    grad!(buffer, risk.model, risk.X, w, risk.Y, ŷ)
end

function value_fun(risk::RiskFunctional, w0::AbstractArray)
    ŷ = value(risk.model.predictor, risk.X, w0)
    function _value(w::AbstractArray)
        value!(ŷ, risk, w)
    end
    _value
end

function grad_fun(risk::RiskFunctional, w0::AbstractVector)
    ŷ = value(risk.model.predictor, risk.X, w0)
    buffer = zeros(length(w0), 1)
    function _grad!(w::AbstractArray, storage::AbstractArray)
        value!(ŷ, risk.model.predictor, risk.X, w)
        grad!(buffer, risk, w, ŷ)
        copy!(storage, buffer)
    end
    _grad!
end

function value_grad_fun(risk::RiskFunctional, w0::AbstractVector)
    ŷ = value(risk.model.predictor, risk.X, w0)
    buffer = zeros(length(w0), 1)
    function _val_grad!(w::AbstractArray, storage::AbstractArray)
        val = value!(ŷ, risk, w)
        grad!(buffer, risk, w, ŷ)
        copy!(storage, buffer)
        val
    end
    _val_grad!
end
