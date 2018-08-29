```@meta
DocTestSetup = quote
    using LossFunctions
end
```

# Efficient Sum and Mean

In many situations we are not really that interested in the
individual loss values (or derivatives) of each observation, but
the sum or mean of them; be it weighted or unweighted. For
example, by computing the unweighted mean of the loss for our
training set, we would effectively compute what is known as the
empirical risk. This is usually the quantity (or an important
part of it) that we are interesting in minimizing.

When we say "weighted" or "unweighted", we are referring to
whether we are explicitly specifying the influence of individual
observations on the result. "Weighing" an observation is achieved
by multiplying its value with some number (i.e. the "weight" of
that observation). As a consequence that weighted observation
will have a stronger or weaker influence on the result. In order
to weigh an observation we have to know which array dimension (if
there are more than one) denotes the observations. On the other
hand, for computing an unweighted result we don't actually need
to know anything about the meaning of the array dimensions, as
long as the `targets` and the `outputs` are of compatible
shape and size.

The naive way to compute such an unweighted reduction, would be
to call `mean` or `sum` on the result of the element-wise
operation. The following code snipped show an example of that. We
say "naive", because it will not give us an acceptable
performance.
