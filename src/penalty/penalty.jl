
# ==========================================================================

"No penalty on the coefficients"
immutable NoPenalty <: Penalty end
Base.show(io::IO, p::NoPenalty) = print(io, "NoPenalty")
@inline value{T<:Number}(p::NoPenalty, w::AbstractArray{T}, len::Int=0) = zero(T)
@inline deriv{T<:Number}(p::NoPenalty, wⱼ::T) = zero(T)

@inline function addgrad!{T<:Number}(buffer::AbstractArray{T}, p::NoPenalty, w::AbstractArray, len::Int=length(w))
  buffer
end

# ==========================================================================

"An L1 (LASSO) penalty on the coefficients"
immutable L1Penalty <: Penalty
  λ::Float64
  function L1Penalty(λ::Real)
    λ >= 0 || error("λ must be positive")
    new(Float64(λ))
  end
end
isconvex(::L1Penalty) = true
isdifferentiable(p::L1Penalty) = false
Base.show(io::IO, p::L1Penalty) = print(io, "L1Penalty(λ = $(p.λ))")
function value{T}(p::L1Penalty, w::AbstractArray{T}, len::Int=length(w))
  len_w = length(w)
  @_dimcheck 0 < len <= len_w
  if len == len_w
    p.λ * sumabs(w)
  else
    val = zero(T)
    for i = 1:len
      @inbounds val += abs(w[i])
    end
    val *= p.λ
    val
  end::Float64
end
@inline deriv(p::L1Penalty, wⱼ::Number) = p.λ * sign(wⱼ)

# ==========================================================================

"An L2 (ridge) penalty on the coefficients"
immutable L2Penalty <: Penalty
  λ::Float64
  function L2Penalty(λ::Real)
    λ >= 0 || error("λ must be positive")
    new(Float64(λ))
  end
end
isconvex(::L2Penalty) = true
isdifferentiable(::L2Penalty) = true
Base.show(io::IO, p::L2Penalty) = print(io, "L2Penalty(λ = $(p.λ))")
@inline function value{T}(p::L2Penalty, w::AbstractArray{T}, len::Int=length(w))
  len_w = length(w)
  @_dimcheck 0 < len <= len_w
  if len == len_w
    p.λ/2 * sumabs2(w)
  else
    val = zero(T)
    for i = 1:len
      @inbounds val += abs2(w[i])
    end
    val *= p.λ
    val /= 2
    val
  end::Float64
end
@inline deriv(p::L2Penalty, wⱼ::Number) = p.λ * wⱼ

# ==========================================================================

# "A weighted average of L1 and L2 penalties on the coefficients"
# immutable ElasticNetPenalty <: Penalty
#   λ::Float64
#   α::Float64
#   function ElasticNetPenalty(λ::Real, α::Real)
#     0 <= α <= 1 || error("α must be within [0, 1]")
#     λ >= 0 || error("λ must be positive")
#     new(Float64(λ), Float64(α))
#   end
# end
# Base.show(io::IO, p::ElasticNetPenalty) = print(io, "ElasticNetPenalty(λ = $(p.λ), α = $(p.α))")
# @inline value(p::ElasticNetPenalty, w::AbstractVector) = p.λ * (p.α * sumabs(w) + (1 - p.α) * .5 * sumabs2(w))

# ==========================================================================
