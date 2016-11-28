# LossFunctions

_LossFunctions.jl is a Julia package that provides efficient and
well-tested implementations for a diverse set of loss functions
that are commonly used in Machine Learning._

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](https://img.shields.io/badge/docs-latest-blue.svg?style=flat)](http://lossesjl.readthedocs.io/en/latest/?badge=latest) | [![Pkg Eval 0.5](http://pkg.julialang.org/badges/LossFunctions_0.5.svg)](http://pkg.julialang.org/?pkg=LossFunctions) [![Pkg Eval 0.6](http://pkg.julialang.org/badges/LossFunctions_0.6.svg)](http://pkg.julialang.org/?pkg=LossFunctions) | [![Build Status](https://travis-ci.org/JuliaML/LossFunctions.jl.svg?branch=master)](https://travis-ci.org/JuliaML/LossFunctions.jl) [![Build status](https://ci.appveyor.com/api/projects/status/xbwc2fiel40bajsp?svg=true)](https://ci.appveyor.com/project/Evizero/losses-jl) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/LossFunctions.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/LossFunctions.jl?branch=master) |

## Available Losses

 **Distance-based (Regression)** | **Margin-based (Classification)**
:-------------------------------:|:----------------------------------:
![distance_losses](https://cloud.githubusercontent.com/assets/10854026/20031932/5b2868d0-a380-11e6-92bc-3ee048ecde13.png) | ![margin_losses](https://cloud.githubusercontent.com/assets/10854026/20660787/9aeb2c28-b54b-11e6-91ca-1dbe265d0ff1.png)

Others: `PeriodicLoss`, `PoissonLoss`, `ScaledLoss`

## Example

The following code snippets show a simple "hello world" scenario
of how a `Loss` can be used to compute the element-wise values.

```julia
using LossFunctions

true_targets = [  1,  0, -2]
pred_outputs = [0.5,  1, -1]

value(L2DistLoss(), true_targets, pred_outputs)
```
```
3-element Array{Float64,1}:
 0.25
 1.0
 1.0
```

Alternatively, one can also use the loss like a function

```julia
myloss = L2DistLoss()
myloss(true_targets, pred_outputs) # same result as above
```

The function signatures of `value` also apply to the derivatives.

```julia
deriv(L2DistLoss(), true_targets, pred_outputs)
```
```
3-element Array{Float64,1}:
 -1.0
 2.0
 2.0
```

Additionally, we provide mutating versions of most functions.

```julia
buffer = zeros(3)
deriv!(buffer, L2DistLoss(), true_targets, pred_outputs)
```

If need be, one can also compute the mean- or sum-value efficiently,
without allocating a temporary array.

```julia
# or meanvalue
sumvalue(L2DistLoss(), true_targets, pred_outputs)
```
```
0.75
```

Note that this only shows a small part of the functionality this
package provides. For more information please have a look at
the documentation.

## Documentation

Check out the **[latest documentation](http://lossesjl.readthedocs.io/en/latest/index.html)**

Additionally, you can make use of Julia's native docsystem.
The following example shows how to get additional information
on `HingeLoss` within Julia's REPL:

```julia
?HingeLoss
```

## Installation

This package is registered in `METADATA.jl` and can be installed as usual

```julia
Pkg.add("LossFunctions")
```

## License

This code is free to use under the terms of the MIT license.

