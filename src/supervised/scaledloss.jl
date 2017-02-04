"""
    scaledloss(loss::SupervisedLoss, K)

Returns a version of `loss` that is uniformly scaled by `K`.
This function dispatches on the type of `loss` in order to choose
the appropriate type of scaled loss that will be used as the decorator.
For example, if `typeof(loss) <: DistanceLoss` then the given `loss`
will be boxed into a `ScaledDistanceLoss`.

Note: If `typeof(K) <: Number`, then this method will poison the
type-inference of the calling scope. This is because `K` will be
promoted to a type parameter. For a typestable version use the
following signature: `scaledloss(loss, Val{K})`
"""
function scaledloss end

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
        typealias MyScaled$($(string(KIND))) LossFunctions.$($(string(SCALEDKIND))){My$($(string(KIND))),1.5}
        ```

        This new loss-type can then be instantiated in the same
        manner and with the same parameters as the original unscaled
        loss-type.

        In contrast, in order to only create a `K` times scaled
        instance of some specific loss you can use
        `scaledloss(my$($(lowercase(string(KIND)))), Val{K})`.
        See `?scaledloss` for more information.
        """ ->
        immutable ($SCALEDKIND){L<:$KIND,K} <: $KIND
            loss::L

            ($SCALEDKIND)() = typeof(K) <: Number ? new(L()) : _serror()
            ($SCALEDKIND)(args...) = typeof(K) <: Number ? new(L(args...)) : _serror()
            ($SCALEDKIND)(loss::L) = typeof(K) <: Number ? new(loss) : _serror()
        end

        ($SCALEDKIND){T,K}(loss::T, ::Type{Val{K}}) = ($SCALEDKIND){T,K}(loss)
        ($SCALEDKIND){T}(loss::T, k::Number) = ($SCALEDKIND)(loss, Val{k})
        (*){K}(::Type{Val{K}}, loss::$KIND) = ($SCALEDKIND)(loss, Val{K})
        (*)(k::Number, loss::$KIND) = ($SCALEDKIND)(loss, Val{k})
        scaledloss{T<:$KIND,K}(loss::T, ::Type{Val{K}}) = ($SCALEDKIND)(loss, Val{K})

        @generated ($SCALEDKIND){T,K1,K2}(s::$SCALEDKIND{T,K1}, ::Type{Val{K2}}) = :(($($SCALEDKIND))(s.loss, Val{$(K1*K2)}))
    end
end

"""
    Union{ScaledSupervisedLoss, ScaledMarginLoss, ScaledDistanceLoss}

Mainly intended for dispatch. Look at `?ScaledMarginLoss`,
or `ScaledDistanceLoss` for more information.

use `scaledloss` to create a scaled version of some loss.
"""
typealias ScaledLoss{T,K} Union{ScaledSupervisedLoss{T,K}, ScaledMarginLoss{T,K}, ScaledDistanceLoss{T,K}}

scaledloss(l::Loss, k::Number) = scaledloss(l, Val{k})

for fun in (:value, :deriv, :deriv2)
    @eval @fastmath ($fun){T,K}(l::ScaledLoss{T,K}, num::Number) = K * ($fun)(l.loss, num)
    @eval @fastmath ($fun){T,K}(l::ScaledLoss{T,K}, target::Number, output::Number) = K * ($fun)(l.loss, target, output)
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

