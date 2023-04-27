# Developer Documentation

In this part of the documentation we will discuss some of the
internal design aspects of this library. Consequently, the target
audience of this section and its sub-sections is primarily people
interested in contributing to this package. As such, the
information provided here should be of little to no relevance for
users interested in simply applying the package.

## Abstract Types

We have seen in previous sections, that many families of loss
functions are implemented as immutable types with free
parameters. An example for such a family is the
[`L1EpsilonInsLoss`](@ref), which represents all the
``\epsilon``-insensitive loss-functions for each possible
value of ``\epsilon``.

Aside from these special families, there a handful of more
generic families that between them contain almost all of the loss
functions this package implements. These families are defined as
abstract types in the type tree. Their main purpose is two-fold:

- From an end-user's perspective, they are most useful for
  dispatching on the particular kind of prediction problem that
  they are intended for (regression vs classification).

- Form an implementation perspective, these abstract types allow
  us to implement shared functionality and fall-back methods,
  or even allow for a simpler implementation.

Most of the implemented loss functions fall under the umbrella of
supervised losses. As such, we barely mention other types of
losses anywhere in this documentation.

```@docs
SupervisedLoss
```

There are two interesting sub-families of supervised loss
functions.  One of these families is called distance-based. All
losses that belong to this family are implemented as subtype of
the abstract type [`DistanceLoss`](@ref), which itself is subtype
of [`SupervisedLoss`](@ref).

```@docs
DistanceLoss
```

The second core sub-family of supervised losses is called
margin-based. All loss functions that belong to this family are
implemented as subtype of the abstract type [`MarginLoss`](@ref),
which itself is subtype of [`SupervisedLoss`](@ref).

```@docs
MarginLoss
```

## Shared Interface

Each of the three abstract types listed above serves a purpose
other than dispatch. All losses that belong to the same family
share functionality to some degree.

More interestingly, the abstract types [`DistanceLoss`](@ref) and
[`MarginLoss`](@ref), serve an additional purpose aside from
shared functionality. We have seen in the background section what
it is that makes a loss margin-based or distance-based. Without
repeating the definition let us state that it boils down to the
existence of a *representing function* ``\psi``, which allows to
compute a loss using a unary function instead of a binary one.
Indeed, all the subtypes of [`DistanceLoss`](@ref) and
[`MarginLoss`](@ref) are implemented in the unary form of their
representing function.

### Distance-based Losses

Supervised losses that can be expressed as a univariate function
of `output - target` are referred to as distance-based losses.
Distance-based losses are typically utilized for regression
problems. That said, there are also other losses that are useful
for regression problems that don't fall into this category, such
as the [`PeriodicLoss`](@ref).

### Margin-based Losses

Margin-based losses are supervised losses where the values of the
targets are restricted to be in ``\{1,-1\}``, and which can
be expressed as a univariate function `output * target`.
