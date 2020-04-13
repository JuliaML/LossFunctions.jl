for FUN in (:value, :deriv, :deriv2)
    @eval begin
        # Compute element-wise (returns an array)
        @generated function ($FUN)(
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.None) where {Q,M,T,N}
            quote
                $(Expr(:meta, :inline))
                S = typeof(($($FUN))(loss, one(Q), one(T)))
                ($($FUN)).(Ref(loss), target, output)
            end
        end

        # (mutating) Compute element-wise (returns an array)
        function ($(Symbol(FUN,:!)))(
                buffer::AbstractArray,
                loss::SupervisedLoss,
                target::AbstractArray{Q,M},
                output::AbstractArray{T,N},
                ::AggMode.None) where {Q,M,T,N}
            buffer .= ($FUN).(Ref(loss), target, output)
            buffer
        end
    end

    for KIND in (:MarginLoss, :DistanceLoss)
        @eval begin
            # Compute element-wise (returns an array)
            @inline function ($FUN)(
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AggMode.None) where {T,N}
                S = typeof(($FUN)(loss, one(T)))
                ($FUN).(Ref(loss), numbers)::Array{S,N}
            end

            # (mutating) Compute element-wise (returns an array)
            function ($(Symbol(FUN,:!)))(
                    buffer::AbstractArray,
                    loss::$KIND,
                    numbers::AbstractArray{T,N},
                    ::AggMode.None) where {T,N}
                buffer .= ($FUN).(Ref(loss), numbers)
                buffer
            end
        end
    end
end
