# module ParameterLosses

# import ..LearnBase: value, value!, deriv, deriv!, deriv2, value_deriv, grad, grad!,
#                     value_fun, deriv_fun, deriv2_fun, value_deriv_fun,
#                     addgrad!
# import ..LearnBase: isminimizable, isdifferentiable, istwicedifferentiable,
#                     isconvex, islipschitzcont, islocallylipschitzcont,
#                     isclipable, ismarginbased, issymmetric
# import ..LearnBase: ParameterLoss
# import ..LearnBase: @_dimcheck
# import Base: show, call, print, transpose

# @autocomplete ParameterLosses export
#     NoParameterLoss,
#     L1ParameterLoss,
#     L2ParameterLoss
#     # ElasticNetParameterLoss,
#     # SCADParameterLoss,


# ==========================================================================

"No penalty on the coefficients"
immutable NoParameterLoss <: ParameterLoss end
Base.show(io::IO, p::NoParameterLoss) = print(io, "NoParameterLoss")
@inline value{T<:Number}(p::NoParameterLoss, w::AbstractArray{T}, len::Int=0) = zero(T)
@inline deriv{T<:Number}(p::NoParameterLoss, wⱼ::T) = zero(T)

@inline function addgrad!{T<:Number}(buffer::AbstractArray{T}, p::NoParameterLoss, w::AbstractArray, len::Int=length(w))
    buffer
end

# ==========================================================================

"An L1 (LASSO) penalty on the coefficients"
immutable L1ParameterLoss <: ParameterLoss
    λ::Float64
    function L1ParameterLoss(λ::Real)
        λ >= 0 || error("λ must be positive")
        new(Float64(λ))
    end
end
isconvex(::L1ParameterLoss) = true
isdifferentiable(p::L1ParameterLoss) = false
Base.show(io::IO, p::L1ParameterLoss) = print(io, "L1ParameterLoss(λ = $(p.λ))")
function value{T}(p::L1ParameterLoss, w::AbstractArray{T}, len::Int=length(w))
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
@inline deriv(p::L1ParameterLoss, wⱼ::Number) = p.λ * sign(wⱼ)

# ==========================================================================

"An L2 (ridge) penalty on the coefficients"
immutable L2ParameterLoss <: ParameterLoss
    λ::Float64
    function L2ParameterLoss(λ::Real)
        λ >= 0 || error("λ must be positive")
        new(Float64(λ))
    end
end
isconvex(::L2ParameterLoss) = true
isdifferentiable(::L2ParameterLoss) = true
Base.show(io::IO, p::L2ParameterLoss) = print(io, "L2ParameterLoss(λ = $(p.λ))")
@inline function value{T}(p::L2ParameterLoss, w::AbstractArray{T}, len::Int=length(w))
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
@inline deriv(p::L2ParameterLoss, wⱼ::Number) = p.λ * wⱼ

# ==========================================================================

# "A weighted average of L1 and L2 penalties on the coefficients"
# immutable ElasticNetParameterLoss <: ParameterLoss
#   λ::Float64
#   α::Float64
#   function ElasticNetParameterLoss(λ::Real, α::Real)
#     0 <= α <= 1 || error("α must be within [0, 1]")
#     λ >= 0 || error("λ must be positive")
#     new(Float64(λ), Float64(α))
#   end
# end
# Base.show(io::IO, p::ElasticNetParameterLoss) = print(io, "ElasticNetParameterLoss(λ = $(p.λ), α = $(p.α))")
# @inline value(p::ElasticNetParameterLoss, w::AbstractVector) = p.λ * (p.α * sumabs(w) + (1 - p.α) * .5 * sumabs2(w))

# ==========================================================================


# end # module
