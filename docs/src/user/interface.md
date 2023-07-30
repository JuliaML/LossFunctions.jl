```@meta
DocTestSetup = quote
    using LossFunctions
    using LossFunctions.Traits
end
```

# Working with Losses

Even though they are called loss "functions", this package
implements them as immutable types instead of true Julia
functions. There are good reasons for that. For example it allows
us to specify the properties of loss functions explicitly (e.g.
`isconvex(myloss)`). It also makes for a more consistent API when
it comes to computing the value or the derivative. Some loss
functions even have additional parameters that need to be
specified, such as the ``\epsilon`` in the case of the
``\epsilon``-insensitive loss. Here, types allow for member
variables to hide that information away from the method
signatures.

In order to avoid potential confusions with true Julia functions,
we will refer to "loss functions" as "losses" instead. The
available losses share a common interface for the most part. This
section will provide an overview of the basic functionality that
is available for all the different types of losses. We will
discuss how to create a loss, how to compute its value and
derivative, and how to query its properties.

```@docs
Loss
SupervisedLoss
UnsupervisedLoss
```

## Instantiating a Loss

Losses are immutable types. As such, one has to instantiate one
in order to work with it. For most losses, the constructors do
not expect any parameters.

```jldoctest
julia> L2DistLoss()
L2DistLoss()

julia> HingeLoss()
L1HingeLoss()
```

We just said that we need to instantiate a loss in order to work
with it. One could be inclined to belief, that it would be more
memory-efficient to "pre-allocate" a loss when using it in more
than one place.

```jldoctest
julia> loss = L2DistLoss()
L2DistLoss()

julia> loss(3, 2)
1
```

However, that is a common oversimplification. Because all losses
are immutable types, they can live on the stack and thus do not
come with a heap-allocation overhead.

Even more interesting in the example above, is that for such
losses as [`L2DistLoss`](@ref), which do not have any constructor
parameters or member variables, there is no additional code
executed at all. Such singletons are only used for dispatch and
don't even produce any additional code, which you can observe for
yourself in the code below. As such they are zero-cost
abstractions.

```julia-repl
julia> v1(loss,y,t) = loss(y,t)

julia> v2(y,t) = L2DistLoss()(y,t)

julia> @code_llvm v1(loss, 3, 2)
define i64 @julia_v1_70944(i64, i64) #0 {
top:
  %2 = sub i64 %1, %0
  %3 = mul i64 %2, %2
  ret i64 %3
}

julia> @code_llvm v2(3, 2)
define i64 @julia_v2_70949(i64, i64) #0 {
top:
  %2 = sub i64 %1, %0
  %3 = mul i64 %2, %2
  ret i64 %3
}
```

On the other hand, some types of losses are actually more
comparable to whole families of losses instead of just a single
one. For example, the immutable type [`L1EpsilonInsLoss`](@ref)
has a free parameter ``\epsilon``. Each concrete ``\epsilon``
results in a different concrete loss of the same family of
epsilon-insensitive losses.

```jldoctest
julia> L1EpsilonInsLoss(0.5)
L1EpsilonInsLoss{Float64}(0.5)

julia> L1EpsilonInsLoss(1)
L1EpsilonInsLoss{Float64}(1.0)
```

For such losses that do have parameters, it can make a slight
difference to pre-instantiate a loss. While they will live on the
stack, the constructor usually performs some assertions and
conversion for the given parameter. This can come at a slight
overhead. At the very least it will not produce the same exact
code when pre-instantiated. Still, the fact that they are immutable
makes them very efficient abstractions with little to no
performance overhead, and zero memory allocations on the heap.

## Computing the Values

The first thing we may want to do is compute the loss for some
observation (singular). In fact, all losses are implemented on
single observations under the hood, and are functors.

```jldoctest bcast1
julia> loss = L1DistLoss()
L1DistLoss()

julia> loss.([2,5,-2], [1,2,3])
3-element Vector{Int64}:
 1
 3
 5
```

## Computing the 1st Derivatives

Maybe the more interesting aspect of loss functions are their
derivatives. In fact, most of the popular learning algorithm in
Supervised Learning, such as gradient descent, utilize the
derivatives of the loss in one way or the other during the
training process.

To compute the derivative of some loss we expose the function
[`deriv`](@ref). It may be interesting to note explicitly, that
we always compute the derivative in respect to the predicted
`output`, since we are interested in deducing in which direction
the output should change.

```@docs
deriv
```

## Computing the 2nd Derivatives

Additionally to the first derivative, we also provide the
corresponding methods for the second derivative through the
function [`deriv2`](@ref). Note again, that we always compute the
derivative in respect to the predicted `output`.

```@docs
deriv2
```

## Properties of a Loss

In some situations it can be quite useful to assert certain
properties about a loss-function. One such scenario could be when
implementing an algorithm that requires the loss to be strictly
convex or Lipschitz continuous.
Note that we will only skim over the defintions in most cases. A
good treatment of all of the concepts involved can be found in
either [^BOYD2004] or [^STEINWART2008].

[^BOYD2004]:

    Stephen Boyd and Lieven Vandenberghe. ["Convex Optimization"](https://stanford.edu/~boyd/cvxbook/). Cambridge University Press, 2004.

[^STEINWART2008]:

    Steinwart, Ingo, and Andreas Christmann. ["Support vector machines"](https://www.springer.com/us/book/9780387772417). Springer Science & Business Media, 2008.

This package uses functions to represent individual properties of
a loss. It follows a list of implemented property-functions
defined in [LearnBase.jl](https://github.com/JuliaML/LearnBase.jl).

```@docs
isdistancebased
ismarginbased
isminimizable
isdifferentiable
istwicedifferentiable
isconvex
isstrictlyconvex
isstronglyconvex
isnemitski
isunivfishercons
isfishercons
islipschitzcont
islocallylipschitzcont
isclipable
isclasscalibrated
issymmetric
```
