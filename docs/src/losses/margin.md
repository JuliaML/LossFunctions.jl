# Margin-based Losses

Margin-based loss functions are particularly useful for binary
classification. In contrast to the distance-based losses, these
do not care about the difference between true target and
prediction. Instead they penalize predictions based on how well
they agree with the sign of the target.

This section lists all the subtypes of [`MarginLoss`](@ref)
that are implemented in this package.

## ZeroOneLoss

```@docs
ZeroOneLoss
```

## PerceptronLoss

```@docs
PerceptronLoss
```

## L1HingeLoss

```@docs
L1HingeLoss
```

## SmoothedL1HingeLoss

```@docs
SmoothedL1HingeLoss
```

## ModifiedHuberLoss

```@docs
ModifiedHuberLoss
```

## DWDMarginLoss

```@docs
DWDMarginLoss
```

## L2MarginLoss

```@docs
L2MarginLoss
```

## L2HingeLoss

```@docs
L2HingeLoss
```

## LogitMarginLoss

```@docs
LogitMarginLoss
```

## ExpLoss

```@docs
ExpLoss
```

## SigmoidLoss

```@docs
SigmoidLoss
```


```@raw html
</div>
```
