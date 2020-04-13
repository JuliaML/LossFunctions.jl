for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # Compute the sum (returns a Number)
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.Sum) where {Q,M,T,N}
            bigger = M > N ? :target : :output
            S, B = min(M,N), max(M,N)
            quote
                @nexprs $S (n)->@dimcheck(size(target, n) == size(output, n))
                out = zero(($($FUN))(loss, one(Q), one(T)))
                @inbounds @simd for I in CartesianIndices(size($bigger))
                    @nexprs $B n->(i_n = I[n])
                    out += ($($FUN))(loss, @nref($M,target,i), @nref($N,output,i))
                end
                out
            end
        end

        # Compute the sum per observation (returns a Vector)
        function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                avg::AggMode.Sum,
                obsdim::ObsDim.Constant{O}) where {Q,T,N,O}
            S = typeof(($FUN)(loss, one(Q), one(T)))
            buffer = zeros(S, size(output, O))
            ($(Symbol(FUN,:!)))(buffer, loss, target, output, avg, obsdim)
        end

        # (mutating) Compute the sum per observation (returns a Vector)
        function ($(Symbol(FUN,:!)))(
                buffer::AbstractVector{B},
                loss::SupervisedLoss,
                target::AbstractArray{Q,N},
                output::AbstractArray{T,N},
                ::AggMode.Sum,
                ::ObsDim.Constant{O}) where {B,Q,T,N,O}
            N == 1 && throw(ArgumentError("Sum per observation non sensible for two Vectors. Try omitting the obsdim"))
            O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
            @dimcheck size(target) == size(output)
            @dimcheck length(buffer) == size(output, O)
            fill!(buffer, zero(B))
            @inbounds @simd for I in CartesianIndices(size(output))
                buffer[I[O]] += ($FUN)(loss, target[I], output[I])
            end
            buffer
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin
            # Compute the sum (returns a Number)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AggMode.Sum) where {T,N}
                sum(x -> ($FUN)(loss, x), numbers)
            end

            # Compute the sum per observation (returns a Vector)
            function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    avg::AggMode.Sum,
                    obsdim::ObsDim.Constant{O}) where {T,N,O}
                S = typeof(($FUN)(loss, one(T)))
                buffer = zeros(S, size(numbers, O))
                ($(Symbol(FUN,:!)))(buffer, loss, numbers, avg, obsdim)
            end

            # (mutating) Compute the sum per observation (returns a Vector)
            function ($(Symbol(FUN,:!)))(
                    buffer::AbstractVector{B},
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AggMode.Sum,
                    ::ObsDim.Constant{O}) where {B,T,N,O}
                N == 1 && throw(ArgumentError("Sum per observation non sensible for two Vectors. Try omitting the obsdim"))
                O > N && throw(ArgumentError("The specified obsdim is larger as the available dimensions."))
                @dimcheck length(buffer) == size(numbers, O)
                fill!(buffer, zero(B))
                @inbounds @simd for I in CartesianIndices(size(numbers))
                    buffer[I[O]] += ($FUN)(loss, numbers[I])
                end
                buffer
            end
        end
    end
end
