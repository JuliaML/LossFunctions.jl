module ParamCosts

import ..LearnBase: value, value!, deriv, deriv!, deriv2, value_deriv, grad, grad!,
                    value_fun, deriv_fun, deriv2_fun, value_deriv_fun,
                    addgrad!
import ..LearnBase: isminimizable, isdifferentiable, istwicedifferentiable,
                    isconvex, islipschitzcont, islocallylipschitzcont,
                    isclipable, ismarginbased, issymmetric
import ..LearnBase: ParamCost
import ..LearnBase: @_dimcheck
import Base: show, call, print, transpose

export

    NoParamCost,
    L1ParamCost,
    L2ParamCost
    # ElasticNetParamCost,
    # SCADParamCost,


# ==========================================================================

"No penalty on the coefficients"
immutable NoParamCost <: ParamCost end
Base.show(io::IO, p::NoParamCost) = print(io, "NoParamCost")
@inline value{T<:Number}(p::NoParamCost, w::AbstractArray{T}, len::Int=0) = zero(T)
@inline deriv{T<:Number}(p::NoParamCost, wⱼ::T) = zero(T)

@inline function addgrad!{T<:Number}(buffer::AbstractArray{T}, p::NoParamCost, w::AbstractArray, len::Int=length(w))
    buffer
end

# ==========================================================================

"An L1 (LASSO) penalty on the coefficients"
immutable L1ParamCost <: ParamCost
    λ::Float64
    function L1ParamCost(λ::Real)
        λ >= 0 || error("λ must be positive")
        new(Float64(λ))
    end
end
isconvex(::L1ParamCost) = true
isdifferentiable(p::L1ParamCost) = false
Base.show(io::IO, p::L1ParamCost) = print(io, "L1ParamCost(λ = $(p.λ))")
function value{T}(p::L1ParamCost, w::AbstractArray{T}, len::Int=length(w))
    len_w = length(w)
    @_dimcheck 0 < len <= len_w
    if len == len_w
        p.λ * sumabs(w)
    else
        val = zero(T)
        @simd for i = 1:len
            @inbounds val += abs(w[i])
        end
        val *= p.λ
        val
    end::Float64
end
@inline deriv(p::L1ParamCost, wⱼ::Number) = p.λ * sign(wⱼ)

# ==========================================================================

"An L2 (ridge) penalty on the coefficients"
immutable L2ParamCost <: ParamCost
    λ::Float64
    function L2ParamCost(λ::Real)
        λ >= 0 || error("λ must be positive")
        new(Float64(λ))
    end
end
isconvex(::L2ParamCost) = true
isdifferentiable(::L2ParamCost) = true
Base.show(io::IO, p::L2ParamCost) = print(io, "L2ParamCost(λ = $(p.λ))")
@inline function value{T}(p::L2ParamCost, w::AbstractArray{T}, len::Int=length(w))
    len_w = length(w)
    @_dimcheck 0 < len <= len_w
    if len == len_w
        p.λ/2 * sumabs2(w)
    else
        val = zero(T)
        @simd for i = 1:len
            @inbounds val += abs2(w[i])
        end
        val *= p.λ
        val /= 2
        val
    end::Float64
end
@inline deriv(p::L2ParamCost, wⱼ::Number) = p.λ * wⱼ

# ==========================================================================

# "A weighted average of L1 and L2 penalties on the coefficients"
# immutable ElasticNetParamCost <: ParamCost
#   λ::Float64
#   α::Float64
#   function ElasticNetParamCost(λ::Real, α::Real)
#     0 <= α <= 1 || error("α must be within [0, 1]")
#     λ >= 0 || error("λ must be positive")
#     new(Float64(λ), Float64(α))
#   end
# end
# Base.show(io::IO, p::ElasticNetParamCost) = print(io, "ElasticNetParamCost(λ = $(p.λ), α = $(p.α))")
# @inline value(p::ElasticNetParamCost, w::AbstractVector) = p.λ * (p.α * sumabs(w) + (1 - p.α) * .5 * sumabs2(w))

# ==========================================================================


end # module
