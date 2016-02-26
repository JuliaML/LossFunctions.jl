
type RiskFunctional{TRisk<:EmpiricalRisk,
                    TX<:AbstractArray,
                    TY<:AbstractArray,
                    TYhat<:AbstractArray,
                    TGrat<:AbstractArray}
    risk::TRisk
    X::TX
    Y::TY
    Yhat::TYhat
    grad::TGrat
end

function RiskFunctional{TPred<:Predictor, TLoss<:PredictionLoss, TPen<:ParamCost}(
        risk::EmpiricalRisk{TPred, TLoss, TPen},
        X::AbstractMatrix,
        Y::AbstractVector)
    # TODO: come up with better way to decide the coef size
    w0 = zeros(intercept(risk) ? size(X, 1)+1 : size(X, 1))
    grad = zeros(length(w0), 1)
    ŷ = value(risk.predictor, X, w0)
    RiskFunctional{typeof(risk), typeof(X), typeof(Y), typeof(ŷ), typeof(grad)}(
        risk, X, Y, ŷ, grad)
end

function value(func::RiskFunctional, w::AbstractArray)
    value!(func.Yhat, func.risk, func.X, w, func.Y)
end

function grad(func::RiskFunctional, w::AbstractArray)
    value!(func.Yhat, func.risk.predictor, func.X, w)
    grad!(func.grad, func.risk, func.X, w, func.Y, func.Yhat)
end

function value_grad(func::RiskFunctional, w::AbstractArray)
    val = value!(func.Yhat, func.risk, func.X, w, func.Y)
    grad!(func.grad, func.risk, func.X, w, func.Y, func.Yhat)
    val
end

function value_fun(func::RiskFunctional)
    function _value(w::AbstractArray)
        value(func, w)
    end
    _value
end

function grad_fun(func::RiskFunctional)
    function _grad!(w::AbstractArray, storage::AbstractArray)
        grad(func, w)
        copy!(storage, func.grad)
    end
    _grad!
end

function value_grad_fun(func::RiskFunctional)
    function _val_grad!(w::AbstractArray, storage::AbstractArray)
        val = value_grad(func, w)
        copy!(storage, func.grad)
        val
    end
    _val_grad!
end
