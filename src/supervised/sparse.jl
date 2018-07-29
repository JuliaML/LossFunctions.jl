# Function for sparse arrays
# value!, deriv!
# `output` can have more dimensions than `target`, in which case do broadcasting

# TODO: find way to use normal broadcast for this.
# probably with the new changes in MLMetric and compare modes

@inline function value(loss::MarginLoss, target::AbstractSparseArray, output::AbstractArray)
    buffer = similar(output)
    value!(buffer, loss, target, output)
end

##
@generated function value!(
        buffer::AbstractArray,
        loss::MarginLoss,
        target::AbstractSparseArray{Q,Ti,M},
        output::AbstractArray{T,N}
    ) where {T,N,Q,Ti,M}
    M > N && throw(ArgumentError("target has more dimensions than output; broadcasting not supported in this direction."))
    quote
      @_dimcheck size(buffer) == size(output)
      @nexprs $M (n)->@_dimcheck(size(target,n) == size(output,n))
      zeroQ = zero(Q)
      negQ = Q(-1)
      @simd for I in CartesianIndices(size(output))
          @nexprs $N n->(i_n = I[n])
          tgt = @nref($M,target,i)
          if tgt == zeroQ
              # convention is that zeros in a sparse array are interpreted as negative one
              @inbounds @nref($N,buffer,i) = value(loss, negQ, @nref($N,output,i))
          else
              @inbounds @nref($N,buffer,i) = value(loss, tgt, @nref($N,output,i))
          end
      end
      buffer
    end
end
