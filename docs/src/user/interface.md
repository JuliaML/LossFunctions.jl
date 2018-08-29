```@meta
DocTestSetup = quote
    using LossFunctions
end
```

# Working with Losses

Even though they are called loss "functions", this package
implements them as immutable types instead of true Julia
functions. There are good reasons for that. For example it allows
us to specify the properties of losse functions explicitly (e.g.
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

## Instantiating a Loss

Losses are immutable types. As such, one has to instantiate one
in order to work with it. For most losses, the constructors do
not expect any parameters.

```jldoctest
julia> L2DistLoss()
LPDistLoss{2}()

julia> HingeLoss()
L1HingeLoss()
```

We just said that we need to instantiate a loss in order to work
with it. One could be inclined to belief, that it would be more
memory-efficient to "pre-allocate" a loss when using it in more
than one place.

```jldoctest
julia> loss = L2DistLoss()
LPDistLoss{2}()

julia> value(loss, 2, 3)
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
julia> v1(loss,t,y) = value(loss,t,y)

julia> v2(t,y) = value(L2DistLoss(),t,y)

julia> @code_llvm v1(loss, 2, 3)
define i64 @julia_v1_70944(i64, i64) #0 {
top:
  %2 = sub i64 %1, %0
  %3 = mul i64 %2, %2
  ret i64 %3
}

julia> @code_llvm v2(2, 3)
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
single observations under the hood. The core function to compute
the value of a loss is `value`. We will see throughout the
documentation that this function allows for a lot of different
method signatures to accomplish a variety of tasks.

```@docs
value(::SupervisedLoss, ::Number, ::Number)
```

It may be interesting to note, that this function also supports
broadcasting and all the syntax benefits that come with it. Thus,
it is quite simple to make use of preallocated memory for storing
the element-wise results.

```jldoctest bcast1
julia> value.(L1DistLoss(), [1,2,3], [2,5,-2])
3-element Array{Int64,1}:
 1
 3
 5

julia> buffer = zeros(3); # preallocate a buffer

julia> buffer .= value.(L1DistLoss(), [1.,2,3], [2,5,-2])
3-element Array{Float64,1}:
 1.0
 3.0
 5.0
```

Furthermore, with the loop fusion changes that were introduced in
Julia 0.6, one can also easily weight the influence of each
observation without allocating a temporary array.

```jldoctest bcast1
julia> buffer .= value.(L1DistLoss(), [1.,2,3], [2,5,-2]) .* [2,1,0.5]
3-element Array{Float64,1}:
 2.0
 3.0
 2.5
```

Even though broadcasting is supported, we do expose a vectorized
method natively. This is done mainly for API consistency reasons.
Internally it even uses broadcast itself, but it does provide the
additional benefit of a more reliable type-inference.

```@docs
value(::SupervisedLoss, ::AbstractArray, ::AbstractArray)
```

We also provide a mutating version for the same reasons. It
even utilizes `broadcast!` underneath.
