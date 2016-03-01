

# ---------------------------------------------------------------
# output transformations
# ---------------------------------------------------------------

"""
Sensitivity, or error responsibility, is the computation of how much the input to a transformation impacts the final error signal.

We typically use the symbol δ to represent this value, with the understanding that:

```
    δ = ∂L / ∂x
    
where:
    L = total loss
    x = input to the transformation
```

As an example, in a logistic regression:

```
For input x, we apply an affine transformation, followed by the sigmoid function (σ)
    s = wx + b
    y = σ(s)

We then calculate loss (L) using deviation from a target value.  The sensitivity of the sigmoid transformation is:
    δ = ∂L / ∂s
```
"""
function sensitivity(trans::Transformation, loss::Loss, input::AbstractVector, output::AbstractVector, target::AbstractVector)
    deriv(trans, input) * deriv(loss, target, output)
end

function sensitivity!(buffer::AbstractVector, trans::Transformation, loss::Loss, input::AbstractVector, output::AbstractVector, target::AbstractVector)
    deriv!(buffer, trans, input)
    for i in eachindex(target)
        buffer[i] *= deriv(loss, target[i], output[i])
    end
    buffer
end


function sensitivity(trans::Transformation, loss::Loss, input::AbstractVector, target::AbstractVector)
    sensitivity(trans, loss, input, value(trans, input), target)
end

# ---------------------------------------------------------------
# interal transformations (TODO)
# ---------------------------------------------------------------

function sensitivity{T <: Transformation}(trans::Transformation, next_transformations::AbstractVector{T}, sensitivities::AbstractVector)
    # TODO: backprop sensitivities
end
