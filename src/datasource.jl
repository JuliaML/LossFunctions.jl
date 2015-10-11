abstract DataSource

nobs(source::DataSource) = @_not_implemented
nvar(source::DataSource) = @_not_implemented
features(source::DataSource) = @_not_implemented
features(source::DataSource, offset::Int, length::Int) = @_not_implemented

# labeled sources

abstract LabeledDataSource <: DataSource
abstract InMemoryLabeledDataSource <: LabeledDataSource

groundtruth(source::LabeledDataSource) = @_not_implemented
targets(source::LabeledDataSource) = @_not_implemented
targets(source::LabeledDataSource, offset::Int, length::Int) = @_not_implemented
bias(source::LabeledDataSource) = @_not_implemented
nclasses(source::LabeledDataSource) = @_not_implemented
labels(source::LabeledDataSource) = @_not_implemented
classdistribution(source::LabeledDataSource) = @_not_implemented

# ==========================================================================
# In-memory labeled sources

immutable EncodedInMemoryLabeledDataSource{E<:ClassEncoding,G<:Any,N} <: InMemoryLabeledDataSource
  features::AbstractArray{Float64,2}
  targets::AbstractArray{Float64,N}
  groundtruth::AbstractArray{G,1}
  encoding::E
  bias::Float64

  function EncodedInMemoryLabeledDataSource(
      features::AbstractArray{Float64,2},
      targets::AbstractArray{Float64,1},
      groundtruth::AbstractArray{G,1},
      encoding::E,
      bias::Float64)
    typeof(encoding) <: OneOfKClassEncoding && throw(ArgumentError("Can't have OneOutOfK-Encoding with a vector as target"))
    size(features, 2) == length(targets) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
    new(features, targets, groundtruth, encoding, bias)
  end

  function EncodedInMemoryLabeledDataSource(
      features::AbstractArray{Float64,2},
      targets::AbstractArray{Float64,2},
      groundtruth::AbstractArray{G,1},
      encoding::OneOfKClassEncoding,
      bias::Float64)
    nclasses(encoding) == size(targets, 1) || throw(DimensionMismatch("Targets have to have the same number of rows as the encoding has labels"))
    size(features, 2) == size(targets, 2) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
    new(features, targets, groundtruth, encoding, bias)
  end
end

function EncodedInMemoryLabeledDataSource{E<:ClassEncoding,G<:Any}(
    features::AbstractArray{Float64,2},
    targets::AbstractArray{Float64,1},
    groundtruth::AbstractArray{G,1},
    encoding::E,
    bias::Float64 = 1.)
  EncodedInMemoryLabeledDataSource{E,G,1}(features, targets, groundtruth, encoding, bias)
end

function EncodedInMemoryLabeledDataSource{E<:ClassEncoding}(
    features::AbstractArray{Float64,2},
    targets::AbstractArray{Float64,1},
    encoding::E,
    bias::Float64 = 1.)
  typeof(encoding) <: OneOfKClassEncoding && throw(ArgumentError("Can't have OneOutOfK-Encoding with a vector as target"))
  size(features, 2) == length(targets) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
  EncodedInMemoryLabeledDataSource(features, targets, labeldecode(encoding, targets), encoding, bias)
end

function EncodedInMemoryLabeledDataSource{G<:Any}(
    features::AbstractArray{Float64,2},
    targets::AbstractArray{Float64,2},
    groundtruth::AbstractArray{G,1},
    encoding::OneOfKClassEncoding,
    bias::Float64 = 1.)
  EncodedInMemoryLabeledDataSource{OneOfKClassEncoding,G,2}(features, targets, groundtruth, encoding, bias)
end

function EncodedInMemoryLabeledDataSource(
    features::AbstractArray{Float64,2},
    targets::AbstractArray{Float64,2},
    encoding::OneOfKClassEncoding,
    bias::Float64 = 1.)
  nclasses(encoding) == size(targets, 1) || throw(DimensionMismatch("Targets have to have the same number of rows as the encoding has labels"))
  size(features, 2) == size(targets, 2) || throw(DimensionMismatch("Features and targets have to have the same number of observations"))
  EncodedInMemoryLabeledDataSource(features, targets, labeldecode(encoding, targets), encoding, bias)
end

groundtruth{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  source.groundtruth

nobs{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  size(source.features, 2)

nvar{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  size(source.features, 1)

features{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  source.features

features{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}, offset::Int, length::Int) =
  view(source.features, :, offset:(offset+length-1))

targets{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  source.targets

targets{E<:ClassEncoding,G}(source::EncodedInMemoryLabeledDataSource{E,G,1}, offset::Int, length::Int) =
  view(source.targets, offset:(offset + length - 1))

targets{E<:ClassEncoding,G}(source::EncodedInMemoryLabeledDataSource{E,G,2}, offset::Int, length::Int) =
  view(source.targets, :, offset:(offset + length - 1))

bias{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  source.bias

nclasses{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  nclasses(source.encoding)

labels{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  labels(source.encoding)

classdistribution{E<:ClassEncoding,G,N}(source::EncodedInMemoryLabeledDataSource{E,G,N}) =
  classdistribution(source.encoding, labeldecode(source.encoding, source.targets))

function shuffle!{G}(X::Array{Float64,2}, t::Array{Float64,1}, g::Array{G,1})
  rows = size(X, 1)
  cols = size(X, 2)
  for c = 1:cols
    i = rand(c:cols)
    for r = 1:rows
      @inbounds X[r,c], X[r,i] = X[r,i], X[r,c]
    end
    @inbounds t[c], t[i] = t[i], t[c]
    @inbounds g[c], g[i] = g[i], g[c]
  end
  nothing
end

function shuffle!{G}(X::Array{Float64,2}, t::Array{Float64,2}, g::Array{G,1})
  rows = size(X, 1)
  cols = size(X, 2)
  rowsT = size(t, 1)
  for c = 1:cols
    i = rand(c:cols)
    for r = 1:rows
      @inbounds X[r,c], X[r,i] = X[r,i], X[r,c]
    end
    for r = 1:rowsT
      @inbounds t[r,c], t[r,i] = t[r,i], t[r,c]
    end
    @inbounds g[c], g[i] = g[i], g[c]
  end
  nothing
end

# ==========================================================================
# DataFrame labeled sources

immutable DataFrameLabeledDataSource <: LabeledDataSource
  formula::Formula
  dataFrame::DataFrame
end

# ==========================================================================
# Encode a DataFrameLabeledDataSource

function convert{E<:ClassEncoding}(
    ::Type{EncodedInMemoryLabeledDataSource{E}},
    source::DataFrameLabeledDataSource)
  mf = ModelFrame(source.formula, source.dataFrame)
  mm = ModelMatrix(mf)
  dfBias = in(0, mm.assign)
  X = dfBias ? mm.m[:,2:end]: mm.m
  extBias = dfBias * 1.0
  t = convert(Vector{Float64}, model_response(mf))
  ce = E(t)
  t_enc = labelencode(ce, t)
  datasource(X', t_enc, ce, bias=extBias)
end

# ==========================================================================
# Choose best DataSource for the parameters

function datasource{E<:ClassEncoding, G, N}(
    features::AbstractArray{Float64,2},
    targets::AbstractArray{Float64,N},
    groundtruth::AbstractArray{G,1},
    encoding::E;
    bias::Float64 = 1.)
  EncodedInMemoryLabeledDataSource(features, targets, groundtruth, encoding, bias)
end

function datasource{E<:ClassEncoding, N}(
    features::AbstractArray{Float64,2},
    targets::AbstractArray{Float64,N},
    encoding::E;
    bias::Float64 = 1.)
  EncodedInMemoryLabeledDataSource(features, targets, encoding, bias)
end

function datasource(formula::Formula, data::DataFrame)
  DataFrameLabeledDataSource(formula, data)
end

function datasource{E<:ClassEncoding}(formula::Formula, data::DataFrame, ::Type{E})
  convert(EncodedInMemoryLabeledDataSource{E}, DataFrameLabeledDataSource(formula, data))
end
