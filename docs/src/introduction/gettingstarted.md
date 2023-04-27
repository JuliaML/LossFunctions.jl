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
L : \mathbb{R} \times Y \rightarrow [0,\infty)
```

Such a loss function takes these two variables as input and
returns a value that quantifies how "bad" our prediction is
in comparison to the truth. In other words: the lower the
loss, the better the prediction.

From an implementation perspective, we should point out that all
the concrete loss "functions" that this package provides are
actually defined as immutable types, instead of native Julia
functions. We can compute the value of some type of loss using
the functor interface. Let us start with an example of how
to compute the loss of a single observation (i.e. two numbers).

```julia-repl
#         loss       ŷ    y
julia> L2DistLoss()(0.5, 1.0)
0.25
```

Calling the same function using arrays instead of numbers will
return the element-wise results, and thus basically just serve as
a wrapper for broadcast (which by the way is also supported).

```julia-repl
julia> true_targets = [  1,  0, -2];

julia> pred_outputs = [0.5,  2, -1];

julia> L2DistLoss().(pred_outputs, true_targets)
3-element Vector{Float64}:
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
julia> sum(L2DistLoss(), pred_outputs, true_targets)
5.25

julia> mean(L2DistLoss(), pred_outputs, true_targets)
1.75
```

Aside from these standard unweighted average modes, we also
provide weighted alternatives. These expect a weight-factor for
each observation in the predicted outputs and so allow to give
certain observations a stronger influence over the result.

```julia-repl
julia> sum(L2DistLoss(), pred_outputs, true_targets, [2,1,1], normalize=false)
5.5

julia> mean(L2DistLoss(), pred_outputs, true_targets, [2,1,1], normalize=false)
1.8333333333333333
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
