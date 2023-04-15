# Distance-based Losses

Loss functions that belong to the category "distance-based" are
primarily used in regression problems. They utilize the numeric
difference between the predicted output and the true target as a
proxy variable to quantify the quality of individual predictions.

This section lists all the subtypes of [`DistanceLoss`](@ref)
that are implemented in this package.

## LPDistLoss

```@docs
LPDistLoss
```

## L1DistLoss

```@docs
L1DistLoss
```

## L2DistLoss

```@docs
L2DistLoss
```

## LogitDistLoss

```@docs
LogitDistLoss
```

## HuberLoss

```@docs
HuberLoss
```

## L1EpsilonInsLoss

```@docs
L1EpsilonInsLoss
```

## L2EpsilonInsLoss

```@docs
L2EpsilonInsLoss
```

## PeriodicLoss

```@docs
PeriodicLoss
```

## QuantileLoss

```@docs
QuantileLoss
```

## LogCoshLoss

```@docs
LogCoshLoss
```

!!! note

    You may note that our definition of the QuantileLoss looks
    different to what one usually sees in other literature. The
    reason is that we have to correct for the fact that in our
    case ``r = \hat{y} - y`` instead of
    ``r_{\textrm{usual}} = y - \hat{y}``, which means that
    our definition relates to that in the manner of
    ``r = -1 * r_{\textrm{usual}}``.


```@raw html
</div>
```
