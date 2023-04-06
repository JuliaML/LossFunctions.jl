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