
function safeRound(num)
  if VERSION < v"0.4-"
    iround(num)
  else
    round(Integer, num)
  end
end

function safeFloor(num)
  if VERSION < v"0.4-"
    ifloor(num)
  else
    floor(Integer, num)
  end
end

macro _not_implemented()
  quote
    throw(ArgumentError("Not implemented for the given type"))
  end
end

macro _dimcheck(condition)
  :(($condition) || throw(DimensionMismatch("Dimensions of the parameters don't match")))
end
