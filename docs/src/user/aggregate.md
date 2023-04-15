```@meta
DocTestSetup = quote
    using LossFunctions
end
```

# Efficient Sum and Mean

In many situations we are not really that interested in the
individual loss values (or derivatives) of each observation, but
the sum or mean of them; be it weighted or unweighted. For
example, by computing the unweighted mean of the loss for our
training set, we would effectively compute what is known as the
empirical risk. This is usually the quantity (or an important
part of it) that we are interesting in minimizing.

When we say "weighted" or "unweighted", we are referring to
whether we are explicitly specifying the influence of individual
observations on the result. "Weighing" an observation is achieved
by multiplying its value with some number (i.e. the "weight" of
that observation). As a consequence that weighted observation
will have a stronger or weaker influence on the result. In order
to weigh an observation we have to know which array dimension (if
there are more than one) denotes the observations. On the other
hand, for computing an unweighted result we don't actually need
to know anything about the meaning of the array dimensions, as
long as the `targets` and the `outputs` are of compatible
shape and size.

The naive way to compute such an unweighted reduction, would be
to call `mean` or `sum` on the result of the element-wise
operation. The following code snipped show an example of that. We
say "naive", because it will not give us an acceptable
performance.

```jldoctest
julia> value(L1DistLoss(), [2,5,-2], [1.,2,3])
3-element Vector{Float64}:
 1.0
 3.0
 5.0

julia> sum(value(L1DistLoss(), [2,5,-2], [1.,2,3])) # WARNING: Bad code
9.0
```

This works as expected, but there is a price for it. Before the
sum can be computed, [`value`](@ref) will allocate a temporary
array and fill it with the element-wise results. After that,
`sum` will iterate over this temporary array and accumulate the
values accordingly. Bottom line: we allocate temporary memory
that we don't need in the end and could avoid.

For that reason we provide special methods that compute the
common accumulations efficiently without allocating temporary
arrays. These methods can be invoked using an additional
parameter which specifies how the values should be accumulated /
averaged. The type of this parameter has to be a subtype of
`AggregateMode`.

## Aggregation Modes

Before we discuss these memory-efficient methods, let us briefly
introduce the available aggregation mode types. We provide a number
of different aggregation modes, all of which are contained within
the namespace `AggMode`. An instance of such type can then be
used as additional parameter to [`value`](@ref), [`deriv`](@ref),
and [`deriv2`](@ref), as we will see further down.

It follows a list of available aggregation modes. Each of which with
a short description of what their effect would be when used as an
additional parameter to the functions mentioned above.

```@docs
AggMode.None
AggMode.Sum
AggMode.Mean
AggMode.WeightedSum
AggMode.WeightedMean
```

## Unweighted Sum and Mean

As hinted before, we provide special memory efficient methods for
computing the **sum** or the **mean** of the element-wise (or
broadcasted) results of [`value`](@ref), [`deriv`](@ref), and
[`deriv2`](@ref). These methods avoid the allocation of a
temporary array and instead compute the result directly.

## Weighted Sum and Mean

Up to this point, all the averaging was performed in an
unweighted manner. That means that each observation was treated
as equal and had thus the same potential influence on the result.
In this sub-section we will consider the situations in which we
do want to explicitly specify the influence of each observation
(i.e. we want to weigh them). When we say we "weigh" an
observation, what it effectively boils down to is multiplying the
result for that observation (i.e. the computed loss or
derivative) with some number. This is done for every observation
individually.

To get a better understand of what we are talking about, let us
consider performing a weighting scheme manually. The following
code will compute the loss for three observations, and then
multiply the result of the second observation with the number
`2`, while the other two remains as they are. If we then sum up
the results, we will see that the loss of the second observation
was effectively counted twice.

```jldoctest
julia> result = value.(L1DistLoss(), [2,5,-2], [1.,2,3]) .* [1,2,1]
3-element Vector{Float64}:
 1.0
 6.0
 5.0

julia> sum(result)
12.0
```

The point of weighing observations is to inform the learning
algorithm we are working with, that it is more important to us to
predict some observations correctly than it is for others. So
really, the concrete weight-factor matters less than the ratio
between the different weights. In the example above the second
observation was thus considered twice as important as any of the
other two observations.

In the case of multi-dimensional arrays the process isn't that
simple anymore. In such a scenario, computing the weighted sum
(or weighted mean) can be thought of as having an additional
step. First we either compute the sum or (unweighted) average for
each observation (which results in a vector), and then we compute
the weighted sum of all observations.

The following code snipped demonstrates how to compute the
`AggMode.WeightedSum([2,1])` manually. This is **not** meant as
an example of how to do it, but simply to show what is happening
qualitatively. In this example we assume that we are working in a
multi-variable regression setting, in which our data set has four
observations with two target-variables each.

```jldoctest weight
julia> targets = reshape(1:8, (2, 4)) ./ 8
2×4 Matrix{Float64}:
 0.125  0.375  0.625  0.875
 0.25   0.5    0.75   1.0

julia> outputs = reshape(1:2:16, (2, 4)) ./ 8
2×4 Matrix{Float64}:
 0.125  0.625  1.125  1.625
 0.375  0.875  1.375  1.875

julia> # WARNING: BAD CODE - ONLY FOR ILLUSTRATION

julia> tmp = sum(value.(L1DistLoss(), outputs, targets), dims=2)
2×1 Matrix{Float64}:
 1.5
 2.0

julia> sum(tmp .* [2, 1]) # weigh 1st observation twice as high
5.0
```

To manually compute the result for `AggMode.WeightedMean([2,1])`
we follow a similar approach, but use the normalized weight
vector in the last step.

```jldoctest weight
julia> using Statistics # for access to "mean"

julia> # WARNING: BAD CODE - ONLY FOR ILLUSTRATION

julia> tmp = mean(value.(L1DistLoss(), outputs, targets), dims=2)
2×1 Matrix{Float64}:
 0.375
 0.5

julia> sum(tmp .* [0.6666, 0.3333]) # weigh 1st observation twice as high
0.416625
```

Note that you can specify explicitly if you want to normalize the
weight vector. That option is supported for computing the
weighted sum, as well as for computing the weighted mean. See the
documentation for [`AggMode.WeightedSum`](@ref) and
[`AggMode.WeightedMean`](@ref) for more information.

The code-snippets above are of course very inefficient, because
they allocate (multiple) temporary arrays. We only included them
to demonstrate what is happening in terms of desired result /
effect. For doing those computations efficiently we provide
special methods for [`value`](@ref), [`deriv`](@ref),
[`deriv2`](@ref) and their mutating counterparts.

```jldoctest weight
julia> value(L1DistLoss(), [2,5,-2], [1.,2,3], AggMode.WeightedSum([1,2,1]))
12.0

julia> value(L1DistLoss(), [2,5,-2], [1.,2,3], AggMode.WeightedMean([1,2,1]))
1.0
```

We also provide this functionality for [`deriv`](@ref) and
[`deriv2`](@ref) respectively.

```jldoctest weight
julia> deriv(L2DistLoss(), [2,5,-2], [1.,2,3], AggMode.WeightedSum([1,2,1]))
4.0

julia> deriv(L2DistLoss(), [2,5,-2], [1.,2,3], AggMode.WeightedMean([1,2,1]))
0.3333333333333333
```
