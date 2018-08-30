# LossFunctions

_LossFunctions.jl is a Julia package that provides efficient and
well-tested implementations for a diverse set of loss functions
that are commonly used in Machine Learning._

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaML.github.io/LossFunctions.jl/stable) | [![Pkg Eval 0.6](http://pkg.julialang.org/badges/LossFunctions_0.6.svg)](http://pkg.julialang.org/?pkg=LossFunctions) [![Pkg Eval 0.7](http://pkg.julialang.org/badges/LossFunctions_0.7.svg)](http://pkg.julialang.org/?pkg=LossFunctions) | [![Build Status](https://travis-ci.org/JuliaML/LossFunctions.jl.svg?branch=master)](https://travis-ci.org/JuliaML/LossFunctions.jl) [![Build status](https://ci.appveyor.com/api/projects/status/xbwc2fiel40bajsp?svg=true)](https://ci.appveyor.com/project/Evizero/losses-jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/LossFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/LossFunctions.jl?branch=master) |

## Available Losses

 **Distance-based (Regression)** | **Margin-based (Classification)**
:-------------------------------:|:----------------------------------:
![distance_losses](https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/distance.svg) | ![margin_losses](https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/margin.svg)

Others: `PeriodicLoss`, `PoissonLoss`, `ScaledLoss`,
`WeightedBinaryLoss`

## Introduction

Typically, the loss functions we work with in Machine Learning
fall into the category of supervised losses. These are
multivariate functions of two variables, the **true target** `y`,
which represents the "ground truth" (i.e. correct answer), and
the **predicted output** `ŷ`, which is what our model thinks the
truth is. A supervised loss function takes these two variables as
input and returns a value that quantifies how "bad" our
prediction is in comparison to the truth. In other words: *the
lower the loss, the better the prediction.*

This package provides a considerable amount of carefully
implemented loss functions, as well as an API to query their
properties (e.g. convexity). Furthermore, we expose methods to
compute their values, derivatives, and second derivatives for
single observations as well as arbitrarily sized arrays of
observations. In the case of arrays a user additionally has the
ability to define if and how element-wise results are averaged or
summed over.

## Example

The following code snippets show a simple "hello world" scenario
of how this package can be used to work with loss functions in
various ways.

```julia
using LossFunctions
```

All the concrete loss "functions" that this package provides are
actually defined as immutable types, instead of native Julia
functions. We can compute the value of some type of loss using
the function `value()`. Let us start with an example of how to
compute the loss for a group of three of observations. By default
the loss will be computed element-wise.

```julia
julia> true_targets = [  1,  0, -2];

julia> pred_outputs = [0.5,  2, -1];

julia> value(L2DistLoss(), true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  0.25
#  4.0
#  1.0
```

Alternatively, one can also use an instance of a loss just like
one would use any other Julia function. This can make the code
significantly more readable while not impacting performance, as
it is a zero-cost abstraction (i.e. it compiles down to the same
code).

```julia
julia> loss = L2DistLoss()
# LossFunctions.LPDistLoss{2}()

julia> loss(true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  0.25
#  4.0
#  1.0

julia> loss(1, 0.5f0) # single observation
# 0.25f0
```

If you are not actually interested in the element-wise results
individually, but some accumulation of those (such as mean or
sum), you can additionally specify an average mode. This will
avoid allocating a temporary array and directly compute the
result.

```julia
julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.Sum())
# 5.25

julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.Mean())
# 1.75
```

Aside from these standard unweighted average modes, we also
provide weighted alternatives. These expect a weight-factor for
each observation in the predicted outputs and so allow to give
certain observations a stronger influence over the result.

```julia
julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.WeightedSum([2,1,1]))
# 5.5

julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.WeightedMean([2,1,1]))
# 1.375
```

We do not restrict the targets and outputs to be vectors, but
instead allow them to be arrays of any arbitrary shape. The shape
of an array may or may not have an interpretation that is
relevant for computing the loss. It is possible to explicitly
specify which dimension denotes the observations. This is
particularly useful for multivariate regression where one could
want to accumulate the loss per individual observation.

```julia
julia> A = rand(2,3)
# 2×3 Array{Float64,2}:
#  0.0939946  0.97639   0.568107
#  0.183244   0.854832  0.962534

julia> B = rand(2,3)
# 2×3 Array{Float64,2}:
#  0.0538206  0.77055  0.996922
#  0.598317   0.72043  0.912274

julia> value(L2DistLoss(), A, B, AvgMode.Sum())
# 0.420741920634

julia> value(L2DistLoss(), A, B, AvgMode.Sum(), ObsDim.First())
# 2-element Array{Float64,1}:
#  0.227866
#  0.192876

julia> value(L2DistLoss(), A, B, AvgMode.Sum(), ObsDim.Last())
# 3-element Array{Float64,1}:
#  0.1739
#  0.060434
#  0.186408
```

All these function signatures of `value` also apply for computing
the derivatives using `deriv` and the second derivatives using
`deriv2`.

```julia
julia> deriv(L2DistLoss(), true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  -1.0
#   4.0
#   2.0

julia> deriv2(L2DistLoss(), true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  2.0
#  2.0
#  2.0
```

For computing the first and second derivatives we additionally
expose a convenience syntax which allows for a more math-like
look of the code.

```julia
julia> loss = L2DistLoss()
# LossFunctions.LPDistLoss{2}()

julia> loss'(true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  -1.0
#   4.0
#   2.0

julia> loss''(true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  2.0
#  2.0
#  2.0
```

Additionally, we provide mutating versions for the subset of
methods that return an array. These have the same function
signatures with the only difference of requiring an additional
parameter as the first argument. This variable should always be
the preallocated array that is to be used as storage.

```julia
julia> buffer = zeros(3)
# 3-element Array{Float64,1}:
#  0.0
#  0.0
#  0.0

julia> deriv!(buffer, L2DistLoss(), true_targets, pred_outputs)
# 3-element Array{Float64,1}:
#  -1.0
#   4.0
#   2.0
```

Note that this only shows a small part of the functionality this
package provides. For more information please have a look at
the documentation.

## Documentation

Check out the **[latest documentation](https://JuliaML.github.io/LossFunctions.jl/stable)**

Additionally, you can make use of Julia's native docsystem.
The following example shows how to get additional information
on `HingeLoss` within Julia's REPL:

```julia
?HingeLoss
```
```
search: HingeLoss L2HingeLoss L1HingeLoss SmoothedL1HingeLoss

  L1HingeLoss <: MarginLoss

  The hinge loss linearly penalizes every predicition where the
  resulting agreement a = y⋅ŷ < 1 . It is Lipschitz continuous
  and convex, but not strictly convex.

  L(a) = \max \{ 0, 1 - a \}

  --------------------------------------------------------------------

                Lossfunction                     Derivative
        ┌────────────┬────────────┐      ┌────────────┬────────────┐
      3 │'\.                      │    0 │                  ┌------│
        │  ''_                    │      │                  |      │
        │     \.                  │      │                  |      │
        │       '.                │      │                  |      │
      L │         ''_             │   L' │                  |      │
        │            \.           │      │                  |      │
        │              '.         │      │                  |      │
      0 │                ''_______│   -1 │------------------┘      │
        └────────────┴────────────┘      └────────────┴────────────┘
        -2                        2      -2                        2
                   y ⋅ ŷ                            y ⋅ ŷ
```

## Installation

This package is registered in `METADATA.jl` and can be installed
as usual

```julia
import Pkg
Pkg.add("LossFunctions")
```

## License

This code is free to use under the terms of the MIT license.
