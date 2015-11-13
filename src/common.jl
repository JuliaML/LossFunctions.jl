macro _not_implemented()
  quote
    throw(ArgumentError("Not implemented for the given type"))
  end
end

macro _dimcheck(condition)
  :(($condition) || throw(DimensionMismatch("Dimensions of the parameters don't match")))
end
