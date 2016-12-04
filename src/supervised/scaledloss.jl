"""
    scaledloss(loss::SupervisedLoss, k)

Returns a version of `loss` that is uniformly scaled by `k`.
This function dispatches on the type of `loss` in order to choose
the appropriate type of scaled loss that will be used as the decorator.
For example, if `typeof(loss) <: DistanceLoss` then the given `loss`
will be boxed into a `ScaledDistanceLoss`.

Note: If `typeof(k) <: Number`, then this method will poison the
type-inference of the calling scope. This is because `k` will be
promoted to a type parameter. For a typestable version use the
following signature: `scaledloss(loss, Val{k})`
"""
function scaledloss end

for KIND in (:MarginLoss, :DistanceLoss, :SupervisedLoss)
    SCALEDKIND = Symbol(:Scaled, KIND)
    @eval begin
        @doc """
            $($(string(SCALEDKIND))) <: $($(string(KIND)))

        Can an be used to represent a scaled version for a given loss

        Use `scaledloss(my$($(lowercase(string(KIND)))), Val{k})`
        to create an instance of it.
        See `?scaledloss` for more information.
        """ ->
        immutable ($SCALEDKIND){L<:$KIND,K} <: $KIND
            loss::L

            ($SCALEDKIND)(args...) = typeof(K) <: Number ? new(L(args...)) : error()
            ($SCALEDKIND)(loss::L) = typeof(K) <: Number ? new(loss) : error()
        end

        ($SCALEDKIND){T,K}(loss::T, ::Type{Val{K}}) = ($SCALEDKIND){T,K}(loss)
        ($SCALEDKIND){T}(loss::T, k::Number) = ($SCALEDKIND)(loss, Val{k})
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
    @eval ($fun){T,K}(l::ScaledLoss{T,K}, num::Number) = K .* ($fun)(l.loss, num)
    @eval ($fun){T,K}(l::ScaledLoss{T,K}, target::Number, output::Number) = K .* ($fun)(l.loss, target, output)
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

