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

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LPDistLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LPDistLoss2.svg)
``L(r) = \mid r \mid ^p`` | ``L'(r) = p \cdot r \cdot \mid r \mid ^{p-2}``


## L1DistLoss

```@docs
L1DistLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1DistLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1DistLoss2.svg)
``L(r) = \mid r \mid`` | ``L'(r) = \textrm{sign}(r)``


## L2DistLoss

```@docs
L2DistLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2DistLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2DistLoss2.svg)
``L(r) = \mid r \mid ^2`` | ``L'(r) = 2 r``


## LogitDistLoss

```@docs
LogitDistLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LogitDistLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LogitDistLoss2.svg)
``L(r) = - \ln \frac{4 e^r}{(1 + e^r)^2}`` | ``L'(r) = \tanh \left( \frac{r}{2} \right)``


## HuberLoss

```@docs
HuberLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/HuberLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/HuberLoss2.svg)
``L(r) = \begin{cases} \frac{r^2}{2} & \quad \text{if } \mid r \mid \le \alpha \\ \alpha \mid r \mid - \frac{\alpha^2}{2} & \quad \text{otherwise}\\ \end{cases}`` | ``L'(r) = \begin{cases} r & \quad \text{if } \mid r \mid \le \alpha \\ \alpha \cdot \textrm{sign}(r) & \quad \text{otherwise}\\ \end{cases}``


## L1EpsilonInsLoss

```@docs
L1EpsilonInsLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1EpsilonInsLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1EpsilonInsLoss2.svg)
``L(r) = \max \{ 0, \mid r \mid - \epsilon \}`` | ``L'(r) = \begin{cases} \frac{r}{ \mid r \mid } & \quad \text{if } \epsilon \le \mid r \mid \\ 0 & \quad \text{otherwise}\\ \end{cases}``


## L2EpsilonInsLoss

```@docs
L2EpsilonInsLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2EpsilonInsLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2EpsilonInsLoss2.svg)
``L(r) = \max \{ 0, \mid r \mid - \epsilon \}^2`` | ``L'(r) = \begin{cases} 2 \cdot \textrm{sign}(r) \cdot \left( \mid r \mid - \epsilon \right) & \quad \text{if } \epsilon \le \mid r \mid \\ 0 & \quad \text{otherwise}\\ \end{cases}``


## PeriodicLoss

```@docs
PeriodicLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/PeriodicLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/PeriodicLoss2.svg)
``L(r) = 1 - \cos \left ( \frac{2 r \pi}{c} \right )`` | ``L'(r) = \frac{2 \pi}{c} \cdot \sin \left( \frac{2r \pi}{c} \right)``


## QuantileLoss

```@docs
QuantileLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/QuantileLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/QuantileLoss2.svg)
``L(r) = \begin{cases} \left( 1 - \tau \right) r & \quad \text{if } r \ge 0 \\ - \tau r & \quad \text{otherwise} \\ \end{cases}`` | ``L(r) = \begin{cases} 1 - \tau & \quad \text{if } r \ge 0 \\ - \tau & \quad \text{otherwise} \\ \end{cases}``

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
