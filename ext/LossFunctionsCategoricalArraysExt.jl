module LossFunctionsCategoricalArraysExt

if isdefined(Base, :get_extension)
  import LossFunctions: MisclassLoss, deriv, deriv2
  import CategoricalArrays: CategoricalValue
else
  import ..LossFunctions: MisclassLoss, deriv, deriv2
  import ..CategoricalArrays: CategoricalValue
end

# type alias to make code more readable
const Scalar = Union{Number,CategoricalValue}

(loss::MisclassLoss)(output::Scalar, target::Scalar) = loss(target == output)
deriv(loss::MisclassLoss, output::Scalar, target::Scalar) = deriv(loss, target == output)
deriv2(loss::MisclassLoss, output::Scalar, target::Scalar) = deriv2(loss, target == output)

end
