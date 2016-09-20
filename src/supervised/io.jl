
Base.print(io::IO, loss::SupervisedLoss, args...) = print(io, typeof(loss).name.name, args...)
Base.print(io::IO, loss::L1DistLoss, args...) = print(io, "L1DistLoss", args...)
Base.print(io::IO, loss::L2DistLoss, args...) = print(io, "L2DistLoss", args...)
Base.print{P}(io::IO, loss::LPDistLoss{P}, args...) = print(io, typeof(loss).name.name, " with P=$(P)", args...)
Base.print(io::IO, loss::L1EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with ɛ=$(loss.ε)", args...)
Base.print(io::IO, loss::L2EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with ɛ=$(loss.ε)", args...)
Base.print(io::IO, loss::SmoothedL1HingeLoss, args...) = print(io, typeof(loss).name.name, " with γ = $(loss.gamma)", args...)
Base.print(io::IO, loss::PeriodicLoss, args...) = print(io, typeof(loss).name.name, " with circumf=$(round(loss.k / 2π,1))", args...)
Base.print(io::IO, loss::ScaledLoss, args...) = print(io, typeof(loss).name.name, " $(loss.factor) * [ $(loss.loss) ]", args...)

# -------------------------------------------------------------
# Plot Recipes

_loss_xguide(loss::MarginLoss) = "y ⋅ h(x)"
_loss_xguide(loss::DistanceLoss) = "h(x) - y"

@recipe function plot(loss::SupervisedLoss, xmin = -2, xmax = 2)
    xguide --> _loss_xguide(loss)
    yguide --> "L(y, h(x))"
    label  --> string(loss)
    value_fun(loss), xmin, xmax
end

@recipe function plot{T<:SupervisedLoss}(losses::AbstractVector{T}, xmin = -2, xmax = 2)
    for loss in losses
        @series begin
            loss, xmin, xmax
        end
    end
end

