# ------------------------------------------------------------------
# Licensed under the MIT License. See LICENCE in the project root.
# ------------------------------------------------------------------

Base.print(io::IO, loss::SupervisedLoss, args...) = print(io, typeof(loss).name.name, args...)
Base.print(io::IO, loss::L1DistLoss, args...) = print(io, "L1DistLoss", args...)
Base.print(io::IO, loss::L2DistLoss, args...) = print(io, "L2DistLoss", args...)
Base.print(io::IO, loss::LPDistLoss{P}, args...) where {P} =
  print(io, typeof(loss).name.name, " with P = $(P)", args...)
Base.print(io::IO, loss::L1EpsilonInsLoss, args...) =
  print(io, typeof(loss).name.name, " with \$\\epsilon\$ = $(loss.ε)", args...)
Base.print(io::IO, loss::L2EpsilonInsLoss, args...) =
  print(io, typeof(loss).name.name, " with \$\\epsilon\$ = $(loss.ε)", args...)
Base.print(io::IO, loss::QuantileLoss, args...) =
  print(io, typeof(loss).name.name, " with \$\\tau\$ = $(loss.τ)", args...)
Base.print(io::IO, loss::SmoothedL1HingeLoss, args...) =
  print(io, typeof(loss).name.name, " with \$\\gamma\$ = $(loss.gamma)", args...)
Base.print(io::IO, loss::HuberLoss, args...) =
  print(io, typeof(loss).name.name, " with \$\\alpha\$ = $(loss.d)", args...)
Base.print(io::IO, loss::DWDMarginLoss, args...) = print(io, typeof(loss).name.name, " with q = $(loss.q)", args...)
Base.print(io::IO, loss::PeriodicLoss, args...) =
  print(io, typeof(loss).name.name, " with c = $(round(2π / loss.k, digits=1))", args...)
Base.print(io::IO, loss::ScaledLoss{T,K}, args...) where {T,K} = print(io, "$(K) * ($(loss.loss))", args...)

_round(num) = round(num) == round(num, digits=1) ? round(Int, num) : round(num, digits=1)
function _relation(num)
  if num <= 0
    "negative only"
  elseif num >= 1
    "positive only"
  elseif num < 0.5
    "1:$(_round((1-num)/num)) weighted"
  else
    "$(_round(num/(1-num))):1 weighted"
  end
end
Base.print(io::IO, loss::WeightedMarginLoss{T,W}, args...) where {T,W} =
  print(io, "$(_relation(W)) $(loss.loss)", args...)
