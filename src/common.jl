macro _not_implemented()
    quote
        throw(ArgumentError("Not implemented for the given type"))
    end
end

macro _dimcheck(condition)
    :(($(esc(condition))) || throw(DimensionMismatch("Dimensions of the parameters don't match")))
end

"""
A helper macro to create a const (submod) which mimics a submodule containing the symbols from an export expression (expr).

```
module A
type X end
type Y end
@autocomplete SubMod export X,Y
end

using A

# now you can type A.SubMod.<tab> to autocomplete X and Y
```
"""
macro autocomplete(submod::Symbol, expr::Expr)
    @assert expr.head == :export
    T = gensym("autocomplete")
    t = :(immutable $T end)
#     t.args[3].args = expr.args
    append!(t.args[3].args, expr.args)
#     for sym in expr.args
#         push!(t.args[3].args, sym)
#     end
    c = :(const $submod = $T())
    append!(c.args[1].args[2].args, expr.args)
    esc(quote
        $expr
        $t
        $c
    end)
end

