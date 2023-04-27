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
julia> loss = L1DistLoss()
L1DistLoss()

julia> loss.([2,5,-2], [1.,2,3])
3-element Vector{Float64}:
 1.0
 3.0
 5.0

julia> sum(loss.([2,5,-2], [1.,2,3])) # WARNING: Bad code
9.0
```

This works as expected, but there is a price for it. Before the
sum can be computed, the solution will allocate a temporary
array and fill it with the element-wise results. After that,
`sum` will iterate over this temporary array and accumulate the
values accordingly. Bottom line: we allocate temporary memory
that we don't need in the end and could avoid.

For that reason we provide special methods that compute the
common accumulations efficiently without allocating temporary
arrays.

```jldoctest
julia> sum(L1DistLoss(), [2,5,-2], [1.,2,3])
9.0

julia> mean(L1DistLoss(), [2,5,-2], [1.,2,3])
3.0
```

Up to this point, all the averaging was performed in an
unweighted manner. That means that each observation was treated
as equal and had thus the same potential influence on the result.
In the following we will consider situations in which we
do want to explicitly specify the influence of each observation
(i.e. we want to weigh them). When we say we "weigh" an
observation, what it effectively boils down to is multiplying the
result for that observation (i.e. the computed loss) with some number.
This is done for every observation individually.

To get a better understand of what we are talking about, let us
consider performing a weighting scheme manually. The following
code will compute the loss for three observations, and then
multiply the result of the second observation with the number
`2`, while the other two remains as they are. If we then sum up
the results, we will see that the loss of the second observation
was effectively counted twice.

```jldoctest
julia> result = L1DistLoss().([2,5,-2], [1.,2,3]) .* [1,2,1]
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

```jldoctest weight
julia> sum(L1DistLoss(), [2,5,-2], [1.,2,3], [1,2,1], normalize=false)
12.0

julia> mean(L1DistLoss(), [2,5,-2], [1.,2,3], [1,2,1])
1.0
```