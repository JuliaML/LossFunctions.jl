
Base.print(io::IO, loss::SupervisedLoss, args...) = print(io, typeof(loss).name.name, args...)
Base.print(io::IO, loss::L1DistLoss, args...) = print(io, "L1DistLoss", args...)
Base.print(io::IO, loss::L2DistLoss, args...) = print(io, "L2DistLoss", args...)
Base.print{P}(io::IO, loss::LPDistLoss{P}, args...) = print(io, typeof(loss).name.name, " with P = $(P)", args...)
Base.print(io::IO, loss::L1EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with ɛ = $(loss.eps)", args...)
Base.print(io::IO, loss::L2EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with ɛ = $(loss.eps)", args...)
Base.print(io::IO, loss::SmoothedL1HingeLoss, args...) = print(io, typeof(loss).name.name, " with γ = $(loss.gamma)", args...)

# -------------------------------------------------------------
# Plot Recipes

_loss_xguide(loss::Union{MarginLoss, ZeroOneLoss}) = "y ⋅ h(x)"
_loss_xguide(loss::DistanceLoss) = "h(x) - y"

@recipe function plot(loss::SupervisedLoss, xmin = -2, xmax = 2)
    :xguide --> _loss_xguide(loss)
    :yguide --> "L(y, h(x))"
    :label  --> string(loss)
    value_fun(loss), xmin, xmax
end

@recipe function plot{T<:SupervisedLoss}(losses::AbstractVector{T}, xmin = -2, xmax = 2)
    :yguide --> "L(y, h(x))"
    :label  --> map(string, losses)'
    if issubplot
        :n      --> length(losses)
        :legend --> false
        :title  --> d[:label]
        :xguide --> map(_loss_xguide, losses)'
    else
        :xguide --> _loss_xguide(first(losses))
    end
    map(value_fun, losses), xmin, xmax
end

