# Getting Started

LossFunctions.jl is the result of a collaborative effort to
design and implement an efficient but also convenient-to-use
[Julia](https://julialang.org) library for, well, loss functions.
As such, this package implements the functionality needed to
query various properties about a loss function (such as
convexity), as well as a number of methods to compute its value,
derivative, and second derivative for single observations or
arrays of observations.

In this section we will provide a condensed overview of the
package. In order to keep this overview concise, we will not
discuss any background information or theory on the losses here
in detail.

## Installation

To install
[LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl),
start up Julia and type the following code-snipped into the REPL.
It makes use of the native Julia package manager.

```julia
] add LossFunctions
```

## Overview

Let us take a look at a few examples to get a feeling of how one
can use this library. This package is registered in the Julia
package ecosystem. Once installed the package can be imported
as usual.

```julia
using LossFunctions
```

Typically, the losses we work with in Machine Learning are
multivariate functions of two variables, the **true target**
``y``, which represents the "ground truth" (i.e. correct
answer), and the **predicted output** ``\hat{y}``, which is
what our model thinks the truth is. All losses that can be
expressed in this way will be referred to as supervised losses.
The true targets are often expected to be of a specific set (e.g.
``\{1,-1\}`` in classification), which we will refer to as
``Y``, while the predicted outputs may be any real number.
So for our purposes we can define a supervised loss as follows

```math
L : Y \times \mathbb{R} \rightarrow [0,\infty)
```

Such a loss function takes these two variables as input and
returns a value that quantifies how "bad" our prediction is
in comparison to the truth. In other words: the lower the
loss, the better the prediction.

From an implementation perspective, we should point out that all
the concrete loss "functions" that this package provides are
actually defined as immutable types, instead of native Julia
functions. We can compute the value of some type of loss using
the function [`value`](@ref). Let us start with an example of how
to compute the loss of a single observation (i.e. two numbers).

```julia-repl
#                loss       y    ŷ
julia> value(L2DistLoss(), 1.0, 0.5)
0.25
```

Calling the same function using arrays instead of numbers will
return the element-wise results, and thus basically just serve as
a wrapper for broadcast (which by the way is also supported).

```julia-repl
julia> true_targets = [  1,  0, -2];

julia> pred_outputs = [0.5,  2, -1];

julia> value(L2DistLoss(), true_targets, pred_outputs)
3-element Array{Float64,1}:
 0.25
 4.0
 1.0
```

If you are not actually interested in the element-wise results
individually, but some accumulation of those (such as mean or
sum), you can additionally specify an **aggregation mode**.
This will avoid allocating a temporary array and directly
compute the result.

```julia-repl
julia> value(L2DistLoss(), true_targets, pred_outputs, AggMode.Sum())
5.25

julia> value(L2DistLoss(), true_targets, pred_outputs, AggMode.Mean())
1.75
```

Aside from these standard unweighted average modes, we also
provide weighted alternatives. These expect a weight-factor for
each observation in the predicted outputs and so allow to give
certain observations a stronger influence over the result.

```julia-repl
julia> value(L2DistLoss(), true_targets, pred_outputs, AggMode.WeightedSum([2,1,1]))
5.5

julia> value(L2DistLoss(), true_targets, pred_outputs, AggMode.WeightedMean([2,1,1]))
1.375
```

We do not restrict the targets and outputs to be vectors, but
instead allow them to be arrays of any arbitrary shape. The shape
of an array may or may not have an interpretation that is
relevant for computing the loss. Consequently, those methods that
don't require this information can be invoked using the same
method signature as before, because the results are simply
computed element-wise or accumulated.

```julia-repl
julia> A = rand(2,3)
2×3 Array{Float64,2}:
 0.0939946  0.97639   0.568107
 0.183244   0.854832  0.962534

julia> B = rand(2,3)
2×3 Array{Float64,2}:
 0.0538206  0.77055  0.996922
 0.598317   0.72043  0.912274

julia> value(L2DistLoss(), A, B)
2×3 Array{Float64,2}:
 0.00161395  0.0423701  0.183882
 0.172286    0.0180639  0.00252607

julia> value(L2DistLoss(), A, B, AggMode.Sum())
0.420741920634
```

These methods even allow arrays of different dimensionality, in
which case broadcast is performed. This also applies to computing
the sum and mean, in which case we use custom broadcast
implementations that avoid allocating a temporary array.

```julia-repl
julia> value(L2DistLoss(), rand(2), rand(2,2))
2×2 Array{Float64,2}:
 0.228077  0.597212
 0.789808  0.311914

julia> value(L2DistLoss(), rand(2), rand(2,2), AggMode.Sum())
0.0860658081865589
```

That said, it is possible to explicitly specify which dimension
denotes the observations. This is particularly useful for
multivariate regression where one could want to accumulate the
loss per individual observation.

```julia-repl
julia> value(L2DistLoss(), A, B, AggMode.Sum(), ObsDim.First())
2-element Array{Float64,1}:
 0.227866
 0.192876

julia> value(L2DistLoss(), A, B, AggMode.Sum(), ObsDim.Last())
3-element Array{Float64,1}:
 0.1739
 0.060434
 0.186408

julia> value(L2DistLoss(), A, B, AggMode.WeightedSum([2,1]), ObsDim.First())
0.648608280735
```

All these function signatures of [`value`](@ref) also apply for
computing the derivatives using [`deriv`](@ref) and the second
derivatives using [`deriv2`](@ref).

```julia-repl
julia> true_targets = [  1,  0, -2];

julia> pred_outputs = [0.5,  2, -1];

julia> deriv(L2DistLoss(), true_targets, pred_outputs)
3-element Array{Float64,1}:
 -1.0
  4.0
  2.0

julia> deriv2(L2DistLoss(), true_targets, pred_outputs)
3-element Array{Float64,1}:
 2.0
 2.0
 2.0
```

## Getting Help

To get help on specific functionality you can either look up the
information here, or if you prefer you can make use of Julia's
native doc-system.
The following example shows how to get additional information
on [`L1HingeLoss`](@ref) within Julia's REPL:

```julia
?L1HingeLoss
```

If you find yourself stuck or have other questions concerning the
package you can find us on the Julia's Zulip chat or the *Machine
Learning* domain on Discourse:

- [Machine Learning in Julia](https://discourse.julialang.org/c/domain/ML)

If you encounter a bug or would like to participate in the
further development of this package come find us on Github.

- [JuliaML/LossFunctions.jl](https://github.com/JuliaML/LossFunctions.jl)
