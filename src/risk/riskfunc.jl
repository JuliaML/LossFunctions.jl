
type RiskFunctional{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty, TX<:AbstractArray, TY<:AbstractArray}
    risk::EmpiricalRisk{TPred, TLoss, TPen}
    X::TX
    Y::TY
end

function RiskFunctional{TPred<:Predictor, TLoss<:Loss, TPen<:Penalty}(
        risk::EmpiricalRisk{TPred, TLoss, TPen},
        X::AbstractArray,
        Y::AbstractArray)
    RiskFunctional{TPred, TLoss, TPen, typeof(X), typeof(Y)}(risk, X, Y)
end

function value(func::RiskFunctional, w::AbstractArray, ŷ::AbstractMatrix = value(func.risk.predictor, func.X, w))
    value(func.risk, func.X, w, func.Y, ŷ)
end

function value!(buffer::AbstractMatrix, func::RiskFunctional, w::AbstractArray)
    value!(buffer, func.risk, func.X, w, func.Y)
end

function grad(func::RiskFunctional, w::AbstractArray, ŷ::AbstractMatrix = value(func.risk.predictor, func.X, w))
    grad(func.risk, func.X, w, func.Y, ŷ)
end

function grad!(buffer::AbstractMatrix, func::RiskFunctional, w::AbstractArray, ŷ::AbstractMatrix = value(func.risk.predictor, func.X, w))
    grad!(buffer, func.risk, func.X, w, func.Y, ŷ)
end

function value_fun(func::RiskFunctional, w0::AbstractArray)
    ŷ = value(func.risk.predictor, func.X, w0)
    function _value(w::AbstractArray)
        value!(ŷ, func, w)
    end
    _value
end

function grad_fun(func::RiskFunctional, w0::AbstractVector)
    ŷ = value(func.risk.predictor, func.X, w0)
    buffer = zeros(length(w0), 1)
    function _grad!(w::AbstractArray, storage::AbstractArray)
        value!(ŷ, func.risk.predictor, func.X, w)
        grad!(buffer, func, w, ŷ)
        copy!(storage, buffer)
    end
    _grad!
end

function value_grad_fun(func::RiskFunctional, w0::AbstractVector)
    ŷ = value(func.risk.predictor, func.X, w0)
    buffer = zeros(length(w0), 1)
    function _val_grad!(w::AbstractArray, storage::AbstractArray)
        val = value!(ŷ, func, w)
        grad!(buffer, func, w, ŷ)
        copy!(storage, buffer)
        val
    end
    _val_grad!
end
