# Code based on code from OnlineStats by Josh day (see LICENSE.md)

abstract Penalty

"No penalty on the coefficients"
immutable NoPenalty <: Penalty end
Base.show(io::IO, p::NoPenalty) = print(io, "NoPenalty")
@inline value(p::NoPenalty, w::AbstractVector) = 0.0
@inline prox(wj::Float64, p::NoPenalty, s::Float64) = wj


"An L2 (ridge) penalty on the coefficients"
type L2Penalty <: Penalty
    λ::Float64
    function L2Penalty(λ::Real)
        @assert λ >= 0
        new(Float64(λ))
    end
end
Base.show(io::IO, p::L2Penalty) = print(io, "L2Penalty(λ = $(p.λ))")
@inline value(p::L2Penalty, w::AbstractVector) = sumabs2(w)
# @inline function prox(wj::Float64, p::L2Penalty, s::Float64)
#     wj / (1.0 + s * p.λ)
# end


"An L1 (LASSO) penalty on the coefficients"
type L1Penalty <: Penalty
    λ::Float64
    function L1Penalty(λ::Real)
        @assert λ >= 0
        new(Float64(λ))
    end
end
Base.show(io::IO, p::L1Penalty) = print(io, "L1Penalty(λ = $(p.λ))")
@inline value(p::L1Penalty, w::AbstractVector) = sumabs(w)
@inline prox(wj::Float64, p::L1Penalty, s::Float64) = sign(wj) * max(abs(wj) - s * p.λ, 0.0)


"A weighted average of L1 and L2 penalties on the coefficients"
type ElasticNetPenalty <: Penalty
    λ::Float64
    α::Float64
    function ElasticNetPenalty(λ::Real, α::Real)
        @assert 0 <= α <= 1
        @assert λ >= 0
        new(Float64(λ), Float64(α))
    end
end
Base.show(io::IO, p::ElasticNetPenalty) = print(io, "ElasticNetPenalty(λ = $(p.λ), α = $(p.α))")
@inline value(p::ElasticNetPenalty, w::AbstractVector) = p.λ * (p.α * sumabs(w) + (1 - p.α) * .5 * sumabs2(w))
@inline function prox(wj::Float64, p::ElasticNetPenalty, s::Float64)
    wj = sign(wj) * max(abs(wj) - s * p.λ * p.α, 0.0)  # Lasso prox
    wj = wj / (1.0 + s * p.λ * (1.0 - p.α))            # Ridge prox
end


# http://www.pstat.ucsb.edu/student%20seminar%20doc/SCAD%20Jian%20Shi.pdf
"Smoothly Clipped Absolute Devation penalty on the coefficients"
type SCADPenalty <: Penalty
    λ::Float64
    a::Float64
    function SCADPenalty(λ::Real, a::Real = 3.7)  # 3.7 is what Fan and Li use
        @assert λ >= 0
        @assert a > 2
        new(Float64(λ), Float64(a))
    end
end
Base.show(io::IO, p::SCADPenalty) = print(io, "SCADPenalty(λ = $(p.λ), a = $(p.a))")
@inline function value(p::SCADPenalty, w::AbstractVector)
    val = 0.0
    for j in 1:length(w)
        wj = abs(w[j])
        if wj < p.λ
            val += p.λ * wj
        elseif wj < p.λ * p.a
            val -= 0.5 * (wj^2 - 2.0 * p.a * p.λ * wj + p.λ^2) / (p.a - 1.0)
        else
            val += 0.5 * (p.a + 1.0) * p.λ^2
        end
    end
    return val
end
@inline function prox(wj::Float64, p::SCADPenalty, s::Float64)
    if abs(wj) > p.a * p.λ
    elseif abs(wj) < 2.0 * p.λ
        wj = sign(wj) * max(abs(wj) - s * p.λ, 0.0)
    else
        wj = (wj - s * sign(wj) * p.a * p.λ / (p.a - 1.0)) / (1.0 - (1.0 / p.a - 1.0))
    end
    wj
end


#-----------------------------------------------------------------------# common
Base.copy(p::Penalty) = deepcopy(p)

# Prox operator is only needed for nondifferentiable penalties
# s = step size
@inline function prox!(w::AbstractVector, p::Penalty, s::Float64)
    for j in 1:length(w)
        w[j] = prox(w[j], p, s)
    end
end
