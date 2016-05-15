
Base.print(io::IO, loss::ModelLoss, args...) = print(io, typeof(loss).name.name, args...)
Base.print(io::IO, loss::L1DistLoss, args...) = print(io, "L1DistLoss", args...)
Base.print(io::IO, loss::L2DistLoss, args...) = print(io, "L2DistLoss", args...)
Base.print{P}(io::IO, loss::LPDistLoss{P}, args...) = print(io, typeof(loss).name.name, " with P = $(P)", args...)
Base.print(io::IO, loss::L1EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with ɛ = $(loss.eps)", args...)
Base.print(io::IO, loss::L2EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with ɛ = $(loss.eps)", args...)
Base.print(io::IO, loss::SmoothedL1HingeLoss, args...) = print(io, typeof(loss).name.name, " with γ = $(loss.gamma)", args...)

# -------------------------------------------------------------
# Plot Recipes

_loss_xlabel(loss::Union{MarginLoss, LossFunctions.ZeroOneLoss}) = "y ⋅ h(x)"
_loss_xlabel(loss::DistanceLoss) = "h(x) - y"

@recipe function plot(loss::ModelLoss, xmin = -2, xmax = 2)
    :xlabel --> _loss_xlabel(loss)
    :ylabel --> "L(y, h(x))"
    :label  --> string(loss)
    value_fun(loss), xmin, xmax
end

@recipe function plot{T<:ModelLoss}(losses::AbstractVector{T}, xmin = -2, xmax = 2)
    :ylabel --> "L(y, h(x))"
    :label  --> map(string, losses)'
    if issubplot
        :n      --> length(losses)
        :legend --> false
        :title  --> d[:label]
        :xlabel --> map(_loss_xlabel, losses)'
    else
        :xlabel --> _loss_xlabel(first(losses))
    end
    map(value_fun, losses), xmin, xmax
end

