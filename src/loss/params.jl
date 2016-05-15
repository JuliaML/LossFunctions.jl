"No penalty on the coefficients"
immutable NoParameterLoss <: ParameterLoss end
Base.show(io::IO, loss::NoParameterLoss) = print(io, "NoParameterLoss")
value{T<:Number}(loss::NoParameterLoss, params::AbstractArray{T}, len::Int=0) = zero(T)
deriv{T<:Number}(loss::NoParameterLoss, param::T) = zero(T)

@inline function addgrad!{T<:Number}(buffer::AbstractArray{T}, loss::NoParameterLoss, params::AbstractArray, len::Int=length(params))
    buffer
end

# ============================================================

"An L1 (LASSO) penalty on the coefficients"
immutable L1ParameterLoss <: ParameterLoss
    λ::Float64
    function L1ParameterLoss(λ::Real)
        λ >= 0 || error("λ must be positive")
        new(Float64(λ))
    end
end
isconvex(::L1ParameterLoss) = true
isdifferentiable(loss::L1ParameterLoss) = false
Base.show(io::IO, loss::L1ParameterLoss) = print(io, "L1ParameterLoss(λ = $(loss.λ))")
function value{T}(loss::L1ParameterLoss, params::AbstractArray{T}, len::Int=length(params))
    len_w = length(params)
    @_dimcheck 0 < len <= len_w
    if len == len_w
        loss.λ * sumabs(params)
    else
        val = zero(T)
        @simd for i = 1:len
            @inbounds val += abs(params[i])
        end
        val *= loss.λ
        val
    end::Float64
end
deriv(loss::L1ParameterLoss, param::Number) = loss.λ * sign(param)

# ============================================================

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
Base.show(io::IO, loss::L2ParameterLoss) = print(io, "L2ParameterLoss(λ = $(loss.λ))")
function value{T}(loss::L2ParameterLoss, params::AbstractArray{T}, len::Int=length(params))
    len_w = length(params)
    @_dimcheck 0 < len <= len_w
    if len == len_w
        loss.λ/2 * sumabs2(params)
    else
        val = zero(T)
        @simd for i = 1:len
            @inbounds val += abs2(params[i])
        end
        val *= loss.λ
        val /= 2
        val
    end::Float64
end
deriv(loss::L2ParameterLoss, param::Number) = loss.λ * param

# =============================================================

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
# Base.show(io::IO, loss::ElasticNetParameterLoss) = print(io, "ElasticNetParameterLoss(λ = $(loss.λ), α = $(loss.α))")
# @inline value(loss::ElasticNetParameterLoss, params::AbstractVector) = loss.λ * (loss.α * sumabs(params) + (1 - loss.α) * .5 * sumabs2(params))

# =============================================================
