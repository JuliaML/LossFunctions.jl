for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # Compute the mean (returns a Number)
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.Mean) where {Q,M,T,N}
            bigger = M > N ? :target : :output
            S, B = min(M,N), max(M,N)
            P = promote_type(Q,T)
            quote
                @nexprs $S (n)->@dimcheck(size(target, n) == size(output, n))
                nrm = 1 / $P(length($bigger))
                out = zero(($($FUN))(loss, one(Q), one(T)) * nrm)
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i)) * nrm
                end
                out
            end
        end

        # Compute the mean per observation (returns a Vector)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AggMode.Mean,
                obsdim::ObsDim.Constant{O}) where {Q,T,N,O}
            S = typeof(($FUN)(loss, one(Q), one(T)) / one(Int))
            buffer = zeros(S, size(output, O))
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, obsdim)
        end

        # (mutating) Compute the mean per observation (returns a Vector)
        function ($(Symbol(FUN,:!)))(
                buffer::AbstractVector{B},
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AggMode.Mean,
                ::ObsDim.Constant{O}) where {B,Q,T,N,O}
            N == 1 && throw(ArgumentError("Mean per observation non sensible for two Vectors. Try omitting the obsdim"))
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck length(buffer) == size(output, O)
            fill!(buffer, zero(B))
            P = promote_type(Q,T)
            k = P(prod(size(output,n) for n in 1:N if n != O))
            nrm = 1 / k
            @inbounds @simd for I in CartesianIndices(size(output))
                buffer[I[O]] += ($FUN)(loss, target[I], output[I]) * nrm
            end
            buffer
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin
            # Compute the mean (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AggMode.Mean) where {T,N}
                nrm = 1 / length(numbers)
                mapreduce(x -> ($FUN)(loss, x) * nrm, +, numbers)
            end

            # Compute the mean per observation (returns a Vector)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AggMode.Mean,
                    obsdim::ObsDim.Constant{O}) where {T,N,O}
                S = typeof(($FUN)(loss, one(T)) / one(Int))
                buffer = zeros(S, size(numbers, O))
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, avg, obsdim)
            end

            # (mutating) Compute the mean per observation (returns a Vector)
            function ($(Symbol(FUN,:!)))(
                    buffer::AbstractVector{B},
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AggMode.Mean,
                    ::ObsDim.Constant{O}) where {B,T,N,O}
                N == 1 && throw(ArgumentError("Mean per observation non sensible for two Vectors. Try omitting the obsdim"))
                O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
                @dimcheck length(buffer) == size(numbers, O)
                fill!(buffer, zero(B))
                k = prod(size(numbers,n) for n in 1:N if n != O)::Int
                nrm = 1 / k
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    buffer[I[O]] += ($FUN)(loss, numbers[I]) * nrm
                end
                buffer
            end
        end
    end
end
