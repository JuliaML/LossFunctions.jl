# ===========================================================
# L(y, t) = |y - t|^P

immutable LPDistLoss{P} <: DistanceLoss
    LPDistLoss() = typeof(P) <: Number ? new() : error()
end

LPDistLoss(p::Number) = LPDistLoss{p}()

value{P}(loss::LPDistLoss{P}, difference::Number) = abs(difference)^P
function deriv{P,T<:Number}(loss::LPDistLoss{P}, difference::T)
    if difference == 0
        zero(difference)
    else
        P * difference * abs(difference)^(P-convert(typeof(P), 2))
    end
end
function deriv2{P,T<:Number}(loss::LPDistLoss{P}, difference::T)
    if difference == 0
        zero(difference)
    else
        (abs2(P)-P) * abs(difference)^P / abs2(difference)
    end
end
value_deriv{P}(loss::LPDistLoss{P}, difference::Number) = (value(loss,difference), deriv(loss,difference))

issymmetric{P}(::LPDistLoss{P}) = true
isdifferentiable{P}(::LPDistLoss{P}) = P > 1
isdifferentiable{P}(::LPDistLoss{P}, at) = P > 1 || at != 0
istwicedifferentiable{P}(::LPDistLoss{P}) = P > 1
istwicedifferentiable{P}(::LPDistLoss{P}, at) = P > 1 || at != 0
islipschitzcont{P}(::LPDistLoss{P}) = P == 1
islipschitzcont_deriv{P}(::LPDistLoss{P}) = 1 <= P <= 2
isconvex{P}(::LPDistLoss{P}) = P >= 1
isstronglyconvex{P}(::LPDistLoss{P}) = P > 1

# ===========================================================
# L(y, t) = |y - t|

"""
`L1DistLoss <: DistanceLoss`

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    3 │\\.                     ./│    1 │            ┌------------│
      │ '\\.                 ./' │      │            |            │
      │   \\.               ./   │      │            |            │
      │    '\\.           ./'    │      │_           |           _│
      │      \\.         ./      │      │            |            │
      │       '\\.     ./'       │      │            |            │
      │         \\.   ./         │      │            |            │
    0 │          '\\./'          │   -1 │------------┘            │
      └────────────┴────────────┘      └────────────┴────────────┘
      -3                        3      -3                        3
               h(x) - y                         h(x) - y
"""
typealias L1DistLoss LPDistLoss{1}

sumvalue(loss::L1DistLoss, difference::AbstractArray) = sumabs(difference)
value(loss::L1DistLoss, difference::Number) = abs(difference)
deriv{T<:Number}(loss::L1DistLoss, difference::T) = convert(T, sign(difference))
deriv2{T<:Number}(loss::L1DistLoss, difference::T) = zero(T)
value_deriv(loss::L1DistLoss, difference::Number) = (abs(difference), sign(difference))

isdifferentiable(::L1DistLoss) = false
isdifferentiable(::L1DistLoss, at) = at != 0
istwicedifferentiable(::L1DistLoss) = true
istwicedifferentiable(::L1DistLoss, at) = true
islipschitzcont(::L1DistLoss) = true
islipschitzcont_deriv(::L1DistLoss) = true
isconvex(::L1DistLoss) = true
isstronglyconvex(::L1DistLoss) = false

# ===========================================================
# L(y, t) = (y - t)^2

"""
`L2DistLoss <: DistanceLoss`

              Lossfunction                     Derivative
      ┌────────────┬────────────┐      ┌────────────┬────────────┐
    9 │\\                       /│    3 │                   .r/   │
      │".                     ."│      │                 .r'     │
      │ ".                   ." │      │              _./'       │
      │  ".                 ."  │      │_           .r/         _│
      │   ".               ."   │      │         _:/'            │
      │    '\\.           ./'    │      │       .r'               │
      │      \\.         ./      │      │     .r'                 │
    0 │        "-.___.-"        │   -3 │  _/r'                   │
      └────────────┴────────────┘      └────────────┴────────────┘
      -3                        3      -2                        2
               h(x) - y                         h(x) - y
"""
typealias L2DistLoss LPDistLoss{2}

sumvalue(loss::L2DistLoss, difference::AbstractArray) = sumabs2(difference)
value(loss::L2DistLoss, difference::Number) = abs2(difference)
deriv{T<:Number}(loss::L2DistLoss, difference::T) = T(2) * difference
deriv2{T<:Number}(loss::L2DistLoss, difference::T) = T(2)
value_deriv{T<:Number}(loss::L2DistLoss, difference::T) = (abs2(difference), T(2) * difference)

isdifferentiable(::L2DistLoss) = true
isdifferentiable(::L2DistLoss, at) = true
istwicedifferentiable(::L2DistLoss) = true
istwicedifferentiable(::L2DistLoss, at) = true
islipschitzcont(::L2DistLoss) = false
islipschitzcont_deriv(::L2DistLoss) = true
isconvex(::L2DistLoss) = true
isstronglyconvex(::L2DistLoss) = true


# ===========================================================
# L(y, t) = 1 - cos((y-t)*2π/c)

"""
`PeriodicLoss <: DistanceLoss`

Measures distance on a circle of specified circumference.
""" 
immutable PeriodicLoss{T<:AbstractFloat} <: DistanceLoss
    k::T   # k = 2π/circumference
    function PeriodicLoss(circ::T)
        circ > 0 || error("circumference should be strictly positive")
        new(convert(T, 2π/circ))
    end
end
function PeriodicLoss{T<:Number}(ε::T=1.0)
    if T <: AbstractFloat
        PeriodicLoss{T}(ε)
    else # cast to Float64
        PeriodicLoss{Float64}(Float64(ε))
    end
end

value{T<:Number}(loss::PeriodicLoss, difference::T) = 1 - cos(difference*loss.k)
deriv{T<:Number}(loss::PeriodicLoss, difference::T) = loss.k * sin(difference*loss.k)
deriv2{T<:Number}(loss::PeriodicLoss, difference::T) = abs2(loss.k) * cos(difference*loss.k)
function value_deriv{T<:Number}(loss::PeriodicLoss, difference::T)
    dk = difference*loss.k
    return 1-cos(dk), loss.k*sin(dk)
end

isdifferentiable(::PeriodicLoss) = true
isdifferentiable(::PeriodicLoss, at) = true
istwicedifferentiable(::PeriodicLoss) = true
istwicedifferentiable(::PeriodicLoss, at) = true
islipschitzcont(::PeriodicLoss) = true
islipschitzcont_deriv(::PeriodicLoss) = true
isconvex(::PeriodicLoss) = false
isstronglyconvex(::PeriodicLoss) = false


# ===========================================================
# L(y, t) = max(0, |y - t| - ɛ)

immutable L1EpsilonInsLoss{T<:AbstractFloat} <: DistanceLoss
    ε::T

    function L1EpsilonInsLoss(ɛ::Number)
        ɛ > 0 || error("ɛ must be strictly positive")
        new(convert(Float64, ɛ))
    end
end
typealias EpsilonInsLoss L1EpsilonInsLoss
function L1EpsilonInsLoss{T<:Number}(ε::T)
    if T <: AbstractFloat
        L1EpsilonInsLoss{T}(ε)
    else # cast to Float64
        L1EpsilonInsLoss{Float64}(Float64(ε))
    end
end

function value{T1,T2<:Number}(loss::L1EpsilonInsLoss{T1}, difference::T2)
    T = promote_type(T1,T2)
    max(zero(T), abs(difference) - loss.ε)
end
function deriv{T1,T2<:Number}(loss::L1EpsilonInsLoss{T1}, difference::T2)
    T = promote_type(T1,T2)
    abs(difference) <= loss.ε ? zero(T) : sign(difference)
end
deriv2{T1,T2<:Number}(loss::L1EpsilonInsLoss{T1}, difference::T2) = zero(promote_type(T1,T2))
function value_deriv{T1,T2<:Number}(loss::L1EpsilonInsLoss{T1}, difference::T2)
    T = promote_type(T1,T2)
    absr = abs(difference)
    absr <= loss.ε ? (zero(T), zero(T)) : (absr - loss.ε, sign(difference))
end

issymmetric(::L1EpsilonInsLoss) = true
isdifferentiable(::L1EpsilonInsLoss) = false
isdifferentiable(loss::L1EpsilonInsLoss, at) = abs(at) != loss.ε
istwicedifferentiable(::L1EpsilonInsLoss) = true
istwicedifferentiable(loss::L1EpsilonInsLoss, at) = abs(at) != loss.ε
islipschitzcont(::L1EpsilonInsLoss) = true
islipschitzcont_deriv(::L1EpsilonInsLoss) = true
isconvex(::L1EpsilonInsLoss) = true
isstronglyconvex(::L1EpsilonInsLoss) = false

# ===========================================================
# L(y, t) = max(0, |y - t| - ɛ)^2

immutable L2EpsilonInsLoss{T<:AbstractFloat} <: DistanceLoss
    ε::T

    function L2EpsilonInsLoss(ɛ::Number)
        ɛ > 0 || error("ɛ must be strictly positive")
        new(convert(T, ɛ))
    end
end
function L2EpsilonInsLoss{T<:Number}(ε::T)
    if T <: AbstractFloat
        L2EpsilonInsLoss{T}(ε)
    else # cast to Float64
        L2EpsilonInsLoss{Float64}(Float64(ε))
    end
end

function value{T1,T2<:Number}(loss::L2EpsilonInsLoss{T1}, difference::T2)
    T = promote_type(T1,T2)
    abs2(max(zero(T), abs(difference) - loss.ε))
end
function deriv{T1,T2<:Number}(loss::L2EpsilonInsLoss{T1}, difference::T2)
    T = promote_type(T1,T2)
    absr = abs(difference)
    absr <= loss.ε ? zero(T) : T(2)*sign(difference)*(absr - loss.ε)
end
function deriv2{T1,T2<:Number}(loss::L2EpsilonInsLoss{T1}, difference::T2)
    T = promote_type(T1,T2)
    abs(difference) <= loss.ε ? zero(T) : T(2)
end
function value_deriv{T}(loss::L2EpsilonInsLoss{T}, difference::Number)
    absr = abs(difference)
    diff = absr - loss.ε
    absr <= loss.ε ? (zero(T), zero(T)) : (abs2(diff), T(2)*sign(difference)*diff)
end

issymmetric(::L2EpsilonInsLoss) = true
isdifferentiable(::L2EpsilonInsLoss) = true
isdifferentiable(::L2EpsilonInsLoss, at) = true
istwicedifferentiable(::L2EpsilonInsLoss) = false
istwicedifferentiable(loss::L2EpsilonInsLoss, at) = abs(at) != loss.ε
islipschitzcont(::L2EpsilonInsLoss) = true
islipschitzcont_deriv(::L2EpsilonInsLoss) = true
isconvex(::L2EpsilonInsLoss) = true
isstronglyconvex(::L2EpsilonInsLoss) = true

# ===========================================================
# L(y, t) = -ln(4 * exp(y - t) / (1 + exp(y - t))²)

immutable LogitDistLoss <: DistanceLoss end

function value(loss::LogitDistLoss, difference::Number)
    er = exp(difference)
    T = typeof(er)
    -log(T(4) * er / abs2(one(T) + er))
end
function deriv{T<:Number}(loss::LogitDistLoss, difference::T)
    tanh(difference / T(2))
end
function deriv2(loss::LogitDistLoss, difference::Number)
    er = exp(difference)
    T = typeof(er)
    T(2)*er / abs2(one(T) + er)
end
function value_deriv(loss::LogitDistLoss, difference::Number)
    er = exp(difference)
    T = typeof(er)
    er1 = one(T) + er
    -log(T(4) * er / abs2(er1)), (er - one(T)) / (er1)
end

issymmetric(::LogitDistLoss) = true
isdifferentiable(::LogitDistLoss) = true
isdifferentiable(::LogitDistLoss, at) = true
istwicedifferentiable(::LogitDistLoss) = true
istwicedifferentiable(::LogitDistLoss, at) = true
islipschitzcont(::LogitDistLoss) = true
islipschitzcont_deriv(::LogitDistLoss) = true
isconvex(::LogitDistLoss) = true
isstronglyconvex(::LogitDistLoss) = true

