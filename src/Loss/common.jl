macro _not_implemented()
  quote
    throw(ArgumentError("Not implemented for the given loss"))
  end
end

sigmoid(x) = 1 / (1 + exp(-x))
