```@meta
DocTestSetup = quote
    using LossFunctions
end
```
```@raw html
<div class="loss-docs">
```

# Margin-based Losses

Margin-based loss functions are particularly useful for binary
classification. In contrast to the distance-based losses, these
do not care about the difference between true target and
prediction. Instead they penalize predictions based on how well
they agree with the sign of the target.

This section lists all the subtypes of [`MarginLoss`](@ref)
that are implemented in this package.

## PerceptronLoss

```@docs
PerceptronLoss
```

Lossfunction | Derivative
-------------|------------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/PerceptronLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/PerceptronLoss2.svg)
``L(a) = \max \{ 0, - a \}`` | ``L'(a) = \begin{cases} -1 & \quad \text{if } a < 0 \\ 0 & \quad \text{otherwise}\\ \end{cases}``


## L1HingeLoss

```@docs
L1HingeLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1HingeLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L1HingeLoss2.svg)
``L(a) = \max \{ 0, 1 - a \}`` | ``L'(a) = \begin{cases} -1 & \quad \text{if } a < 1 \\ 0 & \quad \text{otherwise}\\ \end{cases}``


## SmoothedL1HingeLoss

```@docs
SmoothedL1HingeLoss
```

Lossfunction | Derivative
-------------|------------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/SmoothedL1HingeLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/SmoothedL1HingeLoss2.svg)
``L(a) = \begin{cases} \frac{1}{2 \gamma} \cdot \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge 1 - \gamma \\ 1 - \frac{\gamma}{2} - a & \quad \text{otherwise}\\ \end{cases}`` | ``L'(a) = \begin{cases} - \frac{1}{\gamma} \cdot \max \{ 0, 1 - a \} & \quad \text{if } a \ge 1 - \gamma \\ - 1 & \quad \text{otherwise}\\ \end{cases}``


## ModifiedHuberLoss

```@docs
ModifiedHuberLoss
```

Lossfunction | Derivative
-------------|------------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/ModifiedHuberLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/ModifiedHuberLoss2.svg)
`` L(a) = \begin{cases} \max \{ 0, 1 - a \} ^2 & \quad \text{if } a \ge -1 \\ - 4 a & \quad \text{otherwise}\\ \end{cases}`` | ``L'(a) = \begin{cases} - 2 \cdot \max \{ 0, 1 - a \} & \quad \text{if } a \ge -1 \\ - 4 & \quad \text{otherwise}\\ \end{cases}``


## DWDMarginLoss

```@docs
DWDMarginLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/DWDMarginLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/DWDMarginLoss2.svg)
``L(a) = \begin{cases} 1 - a & \quad \text{if } a \le \frac{q}{q+1} \\ \frac{1}{a^q} \frac{q^q}{(q+1)^{q+1}} & \quad \text{otherwise}\\ \end{cases}`` | ``L'(a) = \begin{cases} - 1 & \quad \text{if } a \le \frac{q}{q+1} \\ - \frac{1}{a^{q+1}} \left( \frac{q}{q+1} \right)^{q+1} & \quad \text{otherwise}\\ \end{cases}``


## L2MarginLoss

```@docs
L2MarginLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2MarginLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2MarginLoss2.svg)
``L(a) = {\left( 1 - a \right)}^2`` | ``L'(a) = 2 \left( a - 1 \right)``


## L2HingeLoss

```@docs
L2HingeLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2HingeLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/L2HingeLoss2.svg)
``L(a) = \max \{ 0, 1 - a \} ^2`` | ``L'(a) = \begin{cases} 2 \left( a - 1 \right) & \quad \text{if } a < 1 \\ 0 & \quad \text{otherwise}\\ \end{cases}``


## LogitMarginLoss

```@docs
LogitMarginLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LogitMarginLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/LogitMarginLoss2.svg)
``L(a) = \ln (1 + e^{-a})`` | ``L'(a) = - \frac{1}{1 + e^a}``


## ExpLoss

```@docs
ExpLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/ExpLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/ExpLoss2.svg)
``L(a) = e^{-a}`` | ``L'(a) = - e^{-a}``


## SigmoidLoss

```@docs
SigmoidLoss
```

Lossfunction | Derivative
-------------|------------
![loss](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/SigmoidLoss1.svg) | ![deriv](https://rawgit.com/JuliaML/FileStorage/master/LossFunctions/SigmoidLoss2.svg)
``L(a) = 1 - \tanh(a)`` | ``L'(a) = - \textrm{sech}^2 (a)``


```@raw html
</div>
```
