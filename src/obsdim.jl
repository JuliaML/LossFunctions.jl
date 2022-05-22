"""
Baseclass for all observation dimensions.
"""
abstract type ObsDimension end

"""
    module ObsDim

Singleton types to define which dimension of some data structure
(e.g. some `Array`) denotes the observations.

- `ObsDim.First()`
- `ObsDim.Last()`
- `ObsDim.Constant(dim)`

Used for efficient dispatching
"""
module ObsDim
    using ..LossFunctions: ObsDimension

    """
    Default value for most functions. Denotes that the concept of
    an observation dimension is not defined for the given data.
    """
    struct Undefined <: ObsDimension end

    """
        ObsDim.Last <: ObsDimension

    Defines that the last dimension denotes the observations
    """
    struct Last <: ObsDimension end

    """
        ObsDim.Constant{DIM} <: ObsDimension

    Defines that the dimension `DIM` denotes the observations
    """
    struct Constant{DIM} <: ObsDimension end
    Constant(dim::Int) = Constant{dim}()

    """
        ObsDim.First <: ObsDimension

    Defines that the first dimension denotes the observations
    """
    const First = Constant{1}
end

Base.convert(::Type{ObsDimension}, dim) = throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
Base.convert(::Type{ObsDimension}, dim::ObsDimension) = dim
Base.convert(::Type{ObsDimension}, ::Nothing) = ObsDim.Undefined()
Base.convert(::Type{ObsDimension}, dim::Int) = ObsDim.Constant(dim)
Base.convert(::Type{ObsDimension}, dim::String) = convert(ObsDimension, Symbol(lowercase(dim)))
Base.convert(::Type{ObsDimension}, dims::Tuple) = map(d->convert(ObsDimension, d), dims)
function Base.convert(::Type{ObsDimension}, dim::Symbol)
    if dim == :first || dim == :begin
        ObsDim.First()
    elseif dim == Symbol("end") || dim == :last
        ObsDim.Last()
    elseif dim == Symbol("nothing") || dim == :none || dim == :null || dim == :na || dim == :undefined
        ObsDim.Undefined()
    else
        throw(ArgumentError("Unknown way to specify a obsdim: $dim"))
    end
end

"""
    default_obsdim(data)

The specify the default obsdim for a specific type of data.
Defaults to `ObsDim.Undefined()`
"""
default_obsdim(data) = ObsDim.Undefined()
default_obsdim(A::AbstractArray) = ObsDim.Last()
default_obsdim(tup::Tuple) = map(default_obsdim, tup)

"""
     datasubset(data, [idx], [obsdim])

Return a lazy subset of the observations in `data` that correspond
to the given `idx`. No data should be copied except of the
indices. Note that `idx` can be of type `Int` or `AbstractVector`.
Both options must be supported by a custom type.

If it makes sense for the type of `data`, `obsdim` can be used
to disptach on which dimension of `data` denotes the observations.
See `?ObsDim`.
"""
function datasubset end
