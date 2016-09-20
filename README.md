# Losses

_Losses.jl is a Julia package that provides efficient and
well-tested implementations for a diverse set of loss functions
that are commonly used in Machine Learning._

| **Package Status** | **Package Evaluator** | **Build Status**  |
|:------------------:|:---------------------:|:-----------------:|
| [![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md) [![Documentation Status](http://readthedocs.org/projects/lossesjl/badge/?version=latest)](http://lossesjl.readthedocs.io/en/latest/?badge=latest) | [![Pkg Eval v5](http://pkg.julialang.org/badges/Losses.5.svg)](http://pkg.julialang.org/?pkg=Losses&ver=0.5) | [![Build Status](https://travis-ci.org/JuliaML/Losses.jl.svg?branch=master)](https://travis-ci.org/JuliaML/Losses.jl) [![Build status](https://ci.appveyor.com/api/projects/status/xbwc2fiel40bajsp/branch/master?svg=true)](https://ci.appveyor.com/project/Evizero/losses-jl/branch/master) [![Coverage Status](https://coveralls.io/repos/github/JuliaML/Losses.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaML/Losses.jl?branch=master) |

## Available Losses

 **Distance-based (Regression)** | **Margin-based (Classification)**
:-------------------------------:|:----------------------------------:
![distance_losses](https://cloud.githubusercontent.com/assets/10854026/17837727/62d856b8-67bb-11e6-9e55-c842712b1edb.png) | ![margin_losses](https://cloud.githubusercontent.com/assets/10854026/17837728/62da0bac-67bb-11e6-92eb-fd5b291cdd8a.png)

Others: `PeriodicLoss`, `PoissonLoss`, `ScaledLoss`

## Example

The following code snippets show a simple "hello world" scenario
of how a `Loss` can be used to compute the element-wise values.

```julia
using Losses

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

The same function signatures also apply to the derivatives.

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
Pkg.add("Losses")
```

## License

This code is free to use under the terms of the MIT license.

