abstract Predictor

# ==========================================================================
# h(x,w) = wᵀx

immutable LinearPredictor{BIAS} <: Predictor  
  function LinearPredictor()
    (typeof(BIAS) <: Bool) || throw(MethodError())
    new()
  end
end

LinearPredictor(bias::Bool) = LinearPredictor{bias}()
LinearPredictor(;bias::Bool = true) = LinearPredictor{bias}()

value(h::LinearPredictor, x::Real, w::Real) = x * w
deriv(h::LinearPredictor, x::Real, w::Real) = x
value(h::LinearPredictor, x::AbstractVector, w::AbstractVector) = dot(x, w)
deriv(h::LinearPredictor, x::AbstractVector, w::AbstractVector) = x

# ==========================================================================
# h(x,w) = 1 / (1 + exp(-f(x,w))

sigmoid(x) = 1 / (1 + exp(-x))

immutable SigmoidPredictor{T<:Predictor} <: Predictor
  f::T
end

SigmoidPredictor(f = LinearPredictor()) = SigmoidPredictor{typeof(f)}(f)
value(h::SigmoidPredictor, x, w) = sigmoid(value(h.f, x, w))
function deriv(h::SigmoidPredictor, x, w)
  z = value(h, x, w)
  z * (1 - z) * deriv(h.f, x, w)
end
