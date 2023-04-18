# LossFunctions.jl

_LossFunctions.jl is a Julia package that provides efficient and
well-tested implementations for a diverse set of loss functions
that are commonly used in Machine Learning._

[![License](https://img.shields.io/badge/license-MIT-blue?style=flat-square)](LICENSE.md)
[![Docs](https://img.shields.io/badge/docs-stable-blue?style=flat-square)](https://JuliaML.github.io/LossFunctions.jl/stable)
[![Build Status](https://img.shields.io/github/actions/workflow/status/JuliaML/LossFunctions.jl/CI.yml?branch=master&style=flat-square")](https://github.com/JuliaML/LossFunctions.jl/actions)
[![Coverage Status](https://img.shields.io/codecov/c/github/JuliaML/LossFunctions.jl?style=flat-square)](https://codecov.io/gh/JuliaML/LossFunctions.jl)

## Available Losses

 **Distance-based (Regression)** | **Margin-based (Classification)**
:-------------------------------:|:----------------------------------:
![distance_losses](https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/distance.svg) | ![margin_losses](https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/margin.svg)

Please consult the [documentation](https://JuliaML.github.io/LossFunctions.jl/stable)
for other losses.

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

Get the latest stable release with Julia's package manager:

```julia
] add LossFunctions
```

## License

This code is free to use under the terms of the MIT license.
