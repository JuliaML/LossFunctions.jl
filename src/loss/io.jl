
print(io::IO, c::Cost, args...) = print(io, typeof(c).name.name, args...)
print{P}(io::IO, l::LPDistLoss{P}, args...) = print(io, typeof(l).name.name, " with P = $(P)", args...)
print(io::IO, l::L1EpsilonInsLoss, args...) = print(io, typeof(l).name.name, " with ɛ = $(l.eps)", args...)
print(io::IO, l::L2EpsilonInsLoss, args...) = print(io, typeof(l).name.name, " with ɛ = $(l.eps)", args...)
print(io::IO, l::SmoothedL1HingeLoss, args...) = print(io, typeof(l).name.name, " with γ = $(l.gamma)", args...)

function show(io::IO, loss::Union{MarginBasedLoss,DistanceBasedLoss,ZeroOneLoss})
  println(io, loss)
  newPlot = lineplot(loss, -3:.05:3, margin = 0, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  ylabel!(newPlot, "")
  xlabel!(newPlot, "")
  annotate!(newPlot, :l, 3, "    ")
  annotate!(newPlot, :r, 3, "L(y,f(x))", :blue)
  annotate!(newPlot, :br, "")
  annotate!(newPlot, :bl, "")
  print(io, newPlot)
  newPlot = lineplot(loss', -3:.05:3, margin = 0, width = 20, height = 5, color = :red, name = " ")
  ylabel!(newPlot, "")
  xlabel!(newPlot, xl)
  annotate!(newPlot, :l, 3, "    ")
  annotate!(newPlot, :r, 3, "L'(y,f(x))", :red)
  print(io, newPlot)
end

function show(io::IO, loss::LogitProbLoss)
  println(io, loss)
  f0(x) = value(loss, 0, x)
  f1(x) = value(loss, 1, x)
  newPlot = lineplot([f0, f1], 0.000001, 0.99999, ylim=[0,8], margin = 1, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  ylabel!(newPlot, "")
  xlabel!(newPlot, "")
  annotate!(newPlot, :r, 2, "y = 0", :blue)
  annotate!(newPlot, :r, 3, "y = 1", :red)
  annotate!(newPlot, :l, 3, "L(y,σ) ")
  annotate!(newPlot, :br, "")
  annotate!(newPlot, :bl, "")
  print(io, newPlot)
  g0(x) = deriv(loss, 0, x)
  g1(x) = deriv(loss, 1, x)
  newPlot = lineplot([g0, g1], 0.000001, 0.99999, ylim=[-1,1], margin = 1, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  ylabel!(newPlot, "")
  xlabel!(newPlot, "σ(z)")
  annotate!(newPlot, :r, 1, "", :blue)
  annotate!(newPlot, :r, 2, "", :red)
  annotate!(newPlot, :l, 3, " ∂L/∂z ")
  print(io, newPlot)
end

function lineplot(
    loss::Union{MarginBasedLoss,ZeroOneLoss},
    args...;
    name = string(typeof(loss).name.name),
    nargs...)
  newPlot = lineplot(repr_fun(loss), args...; name = name, nargs...)
  xlabel!(newPlot, "y * f(x)")
  ylabel!(newPlot, "L(y,f(x))")
end

function lineplot(
    loss::DistanceBasedLoss,
    args...;
    name = string(typeof(loss).name.name),
    nargs...)
  newPlot = lineplot(repr_fun(loss), args...; name = name, nargs...)
  xlabel!(newPlot, "y - f(x)")
  ylabel!(newPlot, "L(y,f(x))")
end

function lineplot(
    loss::LogitProbLoss,
    args...;
    ylim = [0, 5],
    title = string(typeof(loss).name.name),
    nargs...)
  f0(x) = value(loss, 0, x)
  f05(x) = value(loss, 0.5, x)
  f1(x) = value(loss, 1, x)
  newPlot = lineplot([f0, f05, f1], args...; ylim = ylim, title = title, nargs...)
  xlabel!(newPlot, "t = σ(wᵀx)")
  ylabel!(newPlot, "L(y,t)")
  annotate!(newPlot, :r, 1, "y = 0", :blue)
  annotate!(newPlot, :r, 2, "y = 0.5", :red)
  annotate!(newPlot, :r, 3, "y = 1", :yellow)
end

function lineplot(
    loss::LogitProbLoss;
    nargs...)
  lineplot(loss, .000001, .999999; nargs...)
end

function lineplot!{C<:Canvas}(
    plot::Plot{C},
    loss::Union{MarginBasedLoss,DistanceBasedLoss,ZeroOneLoss},
    args...;
    name = string(typeof(loss).name.name),
    nargs...)
  lineplot!(plot, repr_fun(loss), args...; name = name, nargs...)
end

function lineplot{T<:Cost}(
    lossvec::AbstractVector{T}, args...; name = "", nargs...)
  n = length(lossvec)
  @assert n > 0
  newPlot = lineplot(lossvec[1], args...; nargs...)
  for i = 2:n
    lineplot!(newPlot, lossvec[i], args...; nargs...)
  end
  newPlot
end

function lineplot!{C<:Canvas,T<:Cost}(
    plot::Plot{C},
    lossvec::AbstractVector{T},
    args...; nargs...)
  n = length(lossvec)
  @assert n > 0
  for i = 1:n
    lineplot!(plot, lossvec[i], args...; nargs...)
  end
  plot
end
