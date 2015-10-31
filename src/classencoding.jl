abstract ClassEncoding

abstract BinaryClassEncoding <: ClassEncoding
abstract MultinomialClassEncoding <: ClassEncoding

# ==========================================================================

module ClassEncodings

export ZeroOneClassEncoding, SignedClassEncoding, MultivalueClassEncoding, OneOfKClassEncoding, OneHotClassEncoding

using MLBase
using ..LearnBase.ClassEncoding
using ..LearnBase.BinaryClassEncoding
using ..LearnBase.MultinomialClassEncoding

immutable ZeroOneClassEncoding{T} <: BinaryClassEncoding
  labelmap::LabelMap{T}

  function ZeroOneClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels != 2
      throw(ArgumentError("The given target vector must have exactly two classes"))
    end
    new(labelmap)
  end
end

ZeroOneClassEncoding{T}(lm::LabelMap{T}) =
  ZeroOneClassEncoding{T}(lm)

ZeroOneClassEncoding{T}(targets::AbstractVector{T}) =
  ZeroOneClassEncoding{T}(labelmap(targets))

#-----------------------------------------------------------

immutable SignedClassEncoding{T} <: BinaryClassEncoding
  labelmap::LabelMap{T}

  function SignedClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels != 2
      throw(ArgumentError("The given target vector must have exactly two classes"))
    end
    new(labelmap)
  end
end

SignedClassEncoding{T}(lm::LabelMap{T}) =
  SignedClassEncoding{T}(lm)

SignedClassEncoding{T}(targets::AbstractVector{T}) =
  SignedClassEncoding{T}(labelmap(targets))

#-----------------------------------------------------------

immutable MultivalueClassEncoding{T} <: MultinomialClassEncoding
  labelmap::LabelMap{T}
  nlabels::Int
  zeroBased::Bool

  function MultivalueClassEncoding(labelmap::LabelMap{T}, zeroBased = false)
    numLabels = length(labelmap.vs)
    if numLabels < 2
      throw(ArgumentError("The given target vector has less than two classes"))
    end
    new(labelmap, numLabels, zeroBased)
  end
end

MultivalueClassEncoding{T}(lm::LabelMap{T}; zero_based = false) =
  MultivalueClassEncoding{T}(lm, zero_based)

MultivalueClassEncoding{T}(targets::AbstractVector{T}; zero_based = false) =
  MultivalueClassEncoding{T}(labelmap(targets), zero_based)

#-----------------------------------------------------------

immutable OneOfKClassEncoding{T} <: MultinomialClassEncoding
  labelmap::LabelMap{T}
  nlabels::Int

  function OneOfKClassEncoding(labelmap::LabelMap{T})
    numLabels = length(labelmap.vs)
    if numLabels < 2
      throw(ArgumentError("The given target vector has less than two classes"))
    end
    new(labelmap, numLabels)
  end
end

OneOfKClassEncoding{T}(lm::LabelMap{T}) =
  OneOfKClassEncoding{T}(lm)

OneOfKClassEncoding{T}(targets::AbstractVector{T}) =
  OneOfKClassEncoding{T}(labelmap(targets))

end

# ==========================================================================

@reexport using LearnBase.ClassEncodings

typealias OneHotClassEncoding OneOfKClassEncoding

# ==========================================================================

nclasses(ce::BinaryClassEncoding) = 2
nclasses(ce::MultinomialClassEncoding) = ce.nlabels

labels{C<:ClassEncoding}(ce::C) = ce.labelmap.vs

#-----------------------------------------------------------

function groupindices{T}(classEncoding::ClassEncoding, targets::AbstractVector{T})
  groupindices(classEncoding.labelmap, targets)
end

#-----------------------------------------------------------

function classdistribution{T}(labelmap::LabelMap{T}, targets::AbstractVector{T})
  labelmap.vs, map(length, groupindices(labelmap, targets))
end

function classdistribution{T}(classEncoding::ClassEncoding, targets::AbstractVector{T})
  classEncoding.labelmap.vs, map(length, groupindices(classEncoding, targets))
end

# ==========================================================================

function labelencode{T}(classEncoding::ZeroOneClassEncoding{T}, targets::AbstractVector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  float(indicies - 1)
end

function labeldecode{T}(classEncoding::ZeroOneClassEncoding{T}, values::AbstractVector{Float64})
  indicies = round(Integer, values + 1)
  labeldecode(classEncoding.labelmap, indicies)
end

#-----------------------------------------------------------

function labelencode{T}(classEncoding::SignedClassEncoding{T}, targets::AbstractVector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  2(indicies - 1.5)
end

function labeldecode{T}(classEncoding::SignedClassEncoding{T}, values::AbstractVector{Float64})
  indicies = round(Integer, (values / 2.) + 1.5)
  labeldecode(classEncoding.labelmap, indicies)
end

#-----------------------------------------------------------

function labelencode{T}(classEncoding::MultivalueClassEncoding{T}, targets::AbstractVector{T})
  labelencode(classEncoding.labelmap, targets) - classEncoding.zeroBased*1.
end

function labeldecode{T}(classEncoding::MultivalueClassEncoding{T}, values::AbstractVector{Float64})
  indicies = round(Integer, values + classEncoding.zeroBased*1.)
  labeldecode(classEncoding.labelmap, indicies)
end

#-----------------------------------------------------------

function labelencode{T}(classEncoding::OneOfKClassEncoding{T}, targets::AbstractVector{T})
  indicies = labelencode(classEncoding.labelmap, targets)
  convert(Matrix{Float64}, indicatormat(indicies)) # this doesn't work if the indexseq is broken (e.g. [1,2,5])
end

function labeldecode{T}(classEncoding::OneOfKClassEncoding{T}, values::AbstractMatrix{Float64})
  numLabels = classEncoding.nlabels
  indicies = map(safeRound, values' * collect(1:numLabels))
  labeldecode(classEncoding.labelmap, indicies)
end

# ==========================================================================

function getLabelString{T}(labelmap::LabelMap{T})
  labels = labelmap.vs
  c = length(labels)
  if c > 10
    labels = labels[1:10]
    labelString = string(join(labels, ", "), ", ... [TRUNC]")
  else
    labelString = join(labels, ", ")
  end
end

#-----------------------------------------------------------

function show{T}(io::IO, classEncoding::ZeroOneClassEncoding{T})
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        ZeroOneClassEncoding (Binary) to {0, 1}
          .labelmap  ...  encoding for: {$labelString}""")
end

function show{T}(io::IO, classEncoding::SignedClassEncoding{T})
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        SignedClassEncoding (Binary) to {-1, 1}
          .labelmap  ...  encoding for: {$labelString}""")
end

function show{T}(io::IO, classEncoding::MultivalueClassEncoding{T})
  c = classEncoding.nlabels
  zB = classEncoding.zeroBased
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        MultivalueClassEncoding (Multinomial) to range $((1:c) - zB*1)
          .nlabels   ...  $c classes
          .labelmap  ...  encoding for: {$labelString}""")
end

function show{T}(io::IO, classEncoding::OneOfKClassEncoding{T})
  c = classEncoding.nlabels
  labelString = getLabelString(classEncoding.labelmap)
  print(io,
        """
        OneOfKClassEncoding (Multinomial) to one-out-of-$c hot-vector
          .nlabels   ...  $c classes
          .labelmap  ...  encoding for: {$labelString}""")
end
