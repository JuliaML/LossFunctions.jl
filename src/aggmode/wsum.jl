for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # Compute the total weighted sum (returns a Number)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                agg::AggMode.WeightedSum,
                ::ObsDim.Constant{O}) where {Q,T,N,O}
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck size(output, O) == length(agg.weights)
            nrm = agg.normalize ? inv(sum(agg.weights)) : inv(one(sum(agg.weights)))
            out = zero(($FUN)(loss, one(Q), one(T)) * (agg.weights[1] * nrm))
            @inbounds @simd for I in CartesianIndices(size(output))
                out += ($FUN)(loss, target[I], output[I]) * (agg.weights[I[O]] * nrm)
            end
            out
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin
            # Compute the total weighted sum (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    agg::AggMode.WeightedSum,
                    ::ObsDim.Constant{O}) where {T,N,O}
                @dimcheck size(numbers, O) == length(agg.weights)
                nrm = agg.normalize ? inv(sum(agg.weights)) : inv(one(sum(agg.weights)))
                out = zero(($FUN)(loss, one(T)) * (agg.weights[1] * nrm))
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    out += ($FUN)(loss, numbers[I]) * (agg.weights[I[O]] * nrm)
                end
                out
            end
        end
    end
end
