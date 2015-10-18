
print(io::IO, loss::Union{MarginBasedLoss,DistanceBasedLoss}, args...) = print(io, typeof(loss), args...)

function show(io::IO, loss::Union{MarginBasedLoss,DistanceBasedLoss})
  println(io, loss)
  newPlot = lineplot(loss, -3, 3, margin = 2, width = 20, height = 5, name = " ")
  xl = xlabel(newPlot)
  xlabel!(newPlot, "")
  ylabel!(newPlot, "L ")
  annotate!(newPlot, :br, "")
  annotate!(newPlot, :bl, "")
  print(io, newPlot)
  newPlot = lineplot(loss', -3, 3, margin = 1, width = 20, height = 5, color = :red, name = " ")
  ylabel!(newPlot, "L'")
  xlabel!(newPlot, xl)
  print(io, newPlot)
end

function lineplot(
    loss::MarginBasedLoss,
    args...;
    name = string(typeof(loss).name.name),
    nargs...)
  newPlot = lineplot(representing_fun(loss), args...; name = name, nargs...)
  xlabel!(newPlot, "y * f(x)")
  ylabel!(newPlot, "L(y,f(x))")
end

function lineplot(
    loss::DistanceBasedLoss,
    args...;
    name = string(typeof(loss).name.name),
    nargs...)
  newPlot = lineplot(representing_fun(loss), args...; name = name, nargs...)
  xlabel!(newPlot, "y - f(x)")
  ylabel!(newPlot, "L(y,f(x))")
end

function lineplot!(
    plot::Plot,
    loss::Union{MarginBasedLoss,DistanceBasedLoss},
    args...;
    name = string(typeof(loss).name.name),
    nargs...)
  lineplot!(plot, representing_fun(loss), args...; name = name, nargs...)
end

function lineplot{T<:Union{MarginBasedLoss,DistanceBasedLoss}}(
    lossvec::AbstractVector{T}, args...; name = "", nargs...)
  n = length(lossvec)
  @assert n > 0
  newPlot = lineplot(lossvec[1], args...; nargs...)
  for i = 2:n
    lineplot!(newPlot, lossvec[i], args...; nargs...)
  end
  newPlot
end

function lineplot!{T<:Union{MarginBasedLoss,DistanceBasedLoss}}(
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
