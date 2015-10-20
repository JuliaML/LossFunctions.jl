
print(io::IO, c::Cost, args...) = print(io, typeof(c), args...)

function show(io::IO, loss::Union{MarginBasedLoss,DistanceBasedLoss})
  println(io, loss)
  newPlot = lineplot(loss, -3, 3, margin = 2, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  ylabel!(newPlot, "")
  xlabel!(newPlot, "")
  annotate!(newPlot, :r, 3, "L(y,f(x))", :blue)
  annotate!(newPlot, :br, "")
  annotate!(newPlot, :bl, "")
  print(io, newPlot)
  newPlot = lineplot(loss', -3, 3, margin = 1, width = 20, height = 5, color = :red, name = " ")
  ylabel!(newPlot, "")
  xlabel!(newPlot, xl)
  annotate!(newPlot, :r, 3, "L'(y,f(x))", :red)
  print(io, newPlot)
end

function show(io::IO, loss::SigmoidCrossentropyLoss)
  println(io, loss)
  f0(x) = value(loss, 0, x)
  f1(x) = value(loss, 1, x)
  newPlot = lineplot([f0, f1], 0.000001, 0.99999, ylim=[0,10], margin = 1, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  ylabel!(newPlot, "L ")
  xlabel!(newPlot, "")
  annotate!(newPlot, :r, 2, "y = 0", :blue)
  annotate!(newPlot, :r, 3, "y = 1", :red)
  annotate!(newPlot, :br, "")
  annotate!(newPlot, :bl, "")
  print(io, newPlot)
  g0(x) = deriv(loss, 0, x)
  g1(x) = deriv(loss, 1, x)
  newPlot = lineplot([g0, g1], 0.000001, 0.99999, ylim=[-1,1], margin = 1, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  ylabel!(newPlot, "L'")
  xlabel!(newPlot, "σ(x)")
  annotate!(newPlot, :r, 1, "", :blue)
  annotate!(newPlot, :r, 2, "", :red)
  print(io, newPlot)
end

function lineplot(
    loss::MarginBasedLoss,
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

function lineplot!(
    plot::Plot,
    loss::Union{MarginBasedLoss,DistanceBasedLoss},
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

function lineplot!{T<:Cost}(
    plot::Plot,
    lossvec::Vector{T},
    args...; nargs...)
  n = length(lossvec)
  @assert n > 0
  for i = 1:n
    lineplot!(plot, lossvec[i], args...; nargs...)
  end
  plot
end
