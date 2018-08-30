@inline _serror() = throw(ArgumentError("Scale factor K has to be strictly positive."))

for KIND in (:MarginLoss, :DistanceLoss, :SupervisedLoss)
    SCALEDKIND = Symbol(:Scaled, KIND)
    @eval begin
        @doc """
            $($(string(SCALEDKIND))){L,K} <: $($(string(KIND)))

        Can an be used to represent a `K` times scaled version of a
        given type of loss `L`, which must be a subtype of
        `$($(string(KIND)))`.
        For example: To create a typealias for a `1.5` times scaled
        version of some loss `My$($(string(KIND)))`, type:

        ```julia
        const MyScaled$($(string(KIND))) = LossFunctions.$($(string(SCALEDKIND))){My$($(string(KIND))),1.5}
        ```

        This new loss-type can then be instantiated in the same
        manner and with the same parameters as the original unscaled
        loss-type.

        In contrast, in order to only create a `K` times scaled
        instance of some specific loss you can use
        `scaled(my$($(lowercase(string(KIND)))), Val(K))`.
        See `?scaled` for more information.
        """
        struct ($SCALEDKIND){L<:$KIND,K} <: $KIND
            loss::L
            (::Type{($SCALEDKIND){L,K}})(loss::L) where {L<:$KIND, K} = new{L,K}(loss)
        end

        @generated function (::Type{($SCALEDKIND){L,K}})(args...) where {L<:$KIND, K}
            typeof(K) <: Number || _serror()
            :(($($SCALEDKIND)){L,K}(L(args...)))
        end
        ($SCALEDKIND)(loss::T, ::Val{K}) where {T,K} = ($SCALEDKIND){T,K}(loss)
        ($SCALEDKIND)(loss::T, k::Number) where {T} = ($SCALEDKIND)(loss, Val(k))
        (*)(::Val{K}, loss::$KIND) where {K} = ($SCALEDKIND)(loss, Val(K))
        (*)(k::Number, loss::$KIND) = ($SCALEDKIND)(loss, Val(k))
        scaled(loss::T, ::Val{K}) where {T<:$KIND,K} = ($SCALEDKIND)(loss, Val(K))

        @generated ($SCALEDKIND)(s::$SCALEDKIND{T,K1}, ::Val{K2}) where {T,K1,K2} = :(($($SCALEDKIND))(s.loss, Val($(K1*K2))))

    end
end

"""
    Union{ScaledSupervisedLoss, ScaledMarginLoss, ScaledDistanceLoss}

Mainly intended for dispatch. Look at `?ScaledMarginLoss`,
or `ScaledDistanceLoss` for more information.

use `scaled` to create a scaled version of some loss.
"""
const ScaledLoss{T,K} = Union{ScaledSupervisedLoss{T,K}, ScaledMarginLoss{T,K}, ScaledDistanceLoss{T,K}}

"""
    scaled(loss::SupervisedLoss, K)

Returns a version of `loss` that is uniformly scaled by `K`.
This function dispatches on the type of `loss` in order to choose
the appropriate type of scaled loss that will be used as the decorator.
For example, if `typeof(loss) <: DistanceLoss` then the given `loss`
will be boxed into a `ScaledDistanceLoss`.

Note: If `typeof(K) <: Number`, then this method will poison the
type-inference of the calling scope. This is because `K` will be
promoted to a type parameter. For a typestable version use the
following signature: `scaled(loss, Val(K))`
"""
scaled(l::Loss, k::Number) = scaled(l, Val(k))

for fun in (:value, :deriv, :deriv2)
    @eval @fastmath ($fun)(l::ScaledLoss{T,K}, num::Number) where {T,K} = K * ($fun)(l.loss, num)
    @eval @fastmath ($fun)(l::ScaledLoss{T,K}, target::Number, output::Number) where {T,K} = K * ($fun)(l.loss, target, output)
end

for prop in [:isminimizable, :isdifferentiable,
             :istwicedifferentiable,
             :isconvex, :isstrictlyconvex,
             :isstronglyconvex, :isnemitski,
             :isunivfishercons, :isfishercons,
             :islipschitzcont, :islocallylipschitzcont,
             :isclipable, :ismarginbased, :isclasscalibrated,
             :isdistancebased, :issymmetric]
    @eval ($prop)(l::ScaledLoss) = ($prop)(l.loss)
end

for prop_param in (:isdifferentiable, :istwicedifferentiable)
    @eval ($prop_param)(l::ScaledLoss, at) = ($prop_param)(l.loss, at)
end
