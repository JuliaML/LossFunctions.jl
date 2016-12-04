
Base.print(io::IO, loss::SupervisedLoss, args...) = print(io, typeof(loss).name.name, args...)
Base.print(io::IO, loss::L1DistLoss, args...) = print(io, "L1DistLoss", args...)
Base.print(io::IO, loss::L2DistLoss, args...) = print(io, "L2DistLoss", args...)
Base.print{P}(io::IO, loss::LPDistLoss{P}, args...) = print(io, typeof(loss).name.name, " with P = $(P)", args...)
Base.print(io::IO, loss::L1EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with \$\\epsilon\$ = $(loss.ε)", args...)
Base.print(io::IO, loss::L2EpsilonInsLoss, args...) = print(io, typeof(loss).name.name, " with \$\\epsilon\$ = $(loss.ε)", args...)
Base.print(io::IO, loss::QuantileLoss, args...) = print(io, typeof(loss).name.name, " with \$\\tau\$ = $(loss.τ)", args...)
Base.print(io::IO, loss::SmoothedL1HingeLoss, args...) = print(io, typeof(loss).name.name, " with \$\\gamma\$ = $(loss.gamma)", args...)
Base.print(io::IO, loss::HuberLoss, args...) = print(io, typeof(loss).name.name, " with \$\\alpha\$ = $(loss.d)", args...)
Base.print(io::IO, loss::DWDMarginLoss, args...) = print(io, typeof(loss).name.name, " with q = $(loss.q)", args...)
Base.print(io::IO, loss::PeriodicLoss, args...) = print(io, typeof(loss).name.name, " with c = $(round(2π / loss.k,1))", args...)
Base.print{T,K}(io::IO, loss::ScaledLoss{T,K}, args...) = print(io, "$(K) * ($(loss.loss))", args...)

# -------------------------------------------------------------
# Plot Recipes

_loss_xguide(loss::MarginLoss) = "y * h(x)"
_loss_xguide(loss::DistanceLoss) = "h(x) - y"

@recipe function plot(drv::Deriv, rng = -2:0.05:2)
    xguide --> _loss_xguide(drv.loss)
    yguide --> "L'(y, h(x))"
    label  --> string(drv.loss)
    deriv_fun(drv.loss), rng
end

@recipe function plot(loss::SupervisedLoss, rng = -2:0.05:2)
    xguide --> _loss_xguide(loss)
    yguide --> "L(y, h(x))"
    label  --> string(loss)
    value_fun(loss), rng
end

@recipe function plot{T<:Deriv}(derivs::AbstractVector{T}, rng = -2:0.05:2)
    for drv in derivs
        @series begin
            drv, rng
        end
    end
end

@recipe function plot{T<:SupervisedLoss}(losses::AbstractVector{T}, rng = -2:0.05:2)
    for loss in losses
        @series begin
            loss, rng
        end
    end
end

