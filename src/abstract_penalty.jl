# Code based on code from OnlineStats by Josh day (see LICENSE.md)

abstract ParameterLoss

Base.copy(p::ParameterLoss) = deepcopy(p)

@inline grad(p::ParameterLoss, w::AbstractArray, len::Int=length(w)) = grad!(zeros(w), p, w, len)

@inline function grad!{T<:Number}(buffer::AbstractArray{T}, p::ParameterLoss, w::AbstractArray, len::Int=length(w))
    @_dimcheck length(buffer) == length(w)
    @_dimcheck 0 < len <= length(w)
    @inbounds buffer[end] = zero(T)
    @simd for j in 1:len
        @inbounds buffer[j] = deriv(p, w[j])
    end
    buffer
end

@inline function addgrad!{T<:Number}(buffer::AbstractArray{T}, p::ParameterLoss, w::AbstractArray, len::Int=length(w))
    @_dimcheck length(buffer) == length(w)
    @_dimcheck 0 < len <= length(w)
    @simd for j in 1:len
        @inbounds buffer[j] += deriv(p, w[j])
    end
    buffer
end
