Efficient Sum and Mean
=========================

In many cases we may not actually be interested in the individual
loss values or derivatives of the observations, but the sum or
mean of them; be it weighted or unweighted. The naive way to do
accomplish, would be to call ``mean`` or ``sum`` on the result of
the element-wise operation.

.. code-block:: jlcon

   julia> value(L1DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
    1.0
    3.0
    5.0

   # WARNING: Bad code
   julia> sum(value(L1DistLoss(), [1.,2,3], [2,5,-2]))
   9.0

This works as expected, but there is a price for it. Before the
sum can be computed, ``value`` will allocate a temporary array
and fill it with the element-wise results. After that, ``sum``
will iterate over this temporary array and accumulate the values
accordingly. Bottom line: we allocate temporary memory that we
don't need and could avoid.

For that reason we provide special methods that compute the
common accumulations efficiently without allocating temporary
memory. These methods can be invoked using an additional
parameter which specifies how the values should be accumulated /
averaged. The type of this parameter has to be a subtype of
``AverageMode``.

Average Modes
---------------

Before we discuss these memory-efficient methods, let us briefly
introduce the available average mode types. We provide a number
of different averages modes, all of which are contained withing
the namespace ``AvgMode``. An instance of such type can then be
used as additional parameter to :func:`value`, :func:`deriv`, and
:func:`deriv2`, as we will see further down.

.. class:: AvgMode.None

   Used by default. This will cause the element-wise results to
   be returned.

.. class:: AvgMode.Sum

   Causes the method to return the unweighted sum of the
   elements instead of the individual elements. Can be used in
   combination with ``ObsDim``, in which case a vector will be
   returned containing the sum for each observation (useful
   mainly for multivariable regression).

.. class:: AvgMode.Mean

   Causes the method to return the unweighted mean of the
   elements instead of the individual elements. Can be used in
   combination with ``ObsDim``, in which case a vector will be
   returned containing the mean for each observation (useful
   mainly for multivariable regression).

.. class:: AvgMode.WeightedSum

   Causes the method to return the weighted sum of all
   observations. The variable ``weights`` has to be a vector of
   the same length as the number of observations. If ``normalize
   = true``, the values of the weight vector will be normalized
   in such as way that they sum to one, instead of remaining as
   specified.

   .. attribute:: weights

      Vector of weight values that can be used to give certain
      observations a bigger influence on the sum.

      .. code-block:: jlcon

         julia> AvgMode.WeightedSum([1,1,2]); # 3 observations

   .. attribute:: normalize

      Boolean that specifies if the weight vector should be
      transformed in such a way that it sums to one (i.e.
      normalized). This will not mutate the weight vector but
      instead happen on the fly during the accumulation.

      Defaults to ``false``. Setting it to true only really makes
      sense in multivalue-regression, otherwise the result will
      be the same as for :class:`WeightedMean`.

      .. code-block:: jlcon

         julia> AvgMode.WeightedSum([1,1,2], normalize = true);

.. class:: AvgMode.WeightedMean

   Causes the method to return the weighted mean of all
   observations. The variable ``weights`` has to be a vector of
   the same length as the number of observations. If ``normalize
   = true``, the values of the weight vector will be normalized
   in such as way that they sum to one, instead of remaining as
   specified.

   .. attribute:: weights

      Vector of weight values that can be used to give certain
      observations a bigger influence on the mean.

      .. code-block:: jlcon

         julia> AvgMode.WeightedMean([1,1,2]); # 3 observations

   .. attribute:: normalize

      Boolean that specifies if the weight vector should be
      transformed in such a way that it sums to one (i.e.
      normalized). This will not mutate the weight vector but
      instead happen on the fly during the accumulation.

      Defaults to ``true``. Setting it to false only really makes
      sense in multivalue-regression, otherwise the result will
      be the same as for :class:`WeightedSum`.

      .. code-block:: jlcon

         julia> AvgMode.WeightedMean([1,1,2], normalize = false);

Unweighted Sum and Mean
-------------------------

As hinted before, we provide special memory efficient methods for
computing the sum or the mean of the element-wise (or
broadcasted) results of :func:`value`. These methods avoid the
allocation of a temporary array and instead compute the result
directly.

When we say "weighted" or "unweighted" accumulation, we are
referring whether we are explicitly specifying the influence of
individual observations on the result. "Weighing" an observation
is achieved by multiplying its value with some number (i.e. the
"weight" of that observation). This implies that in order to
weigh an observation we have to know which array dimension (if
there are more than one) denotes the observations. On the other
hand, for computing an unweighted result we don't actually need
to know anything about the meaning of the array dimensions, as
long as the ``targets`` and the ``outputs`` are of compatible
shape and size.

.. function:: value(loss, targets, outputs, avgmode) -> Number

   Computes the **unweighted** sum or mean (depending on
   `avgmode`) of the individual values of the loss function for
   each index-pair in `targets` and `outputs`. This method will
   not allocate a temporary array.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()` or
                               :func:`AvgMode.Mean()`
   :return: The unweighted sum or mean of the element-wise values
            of the loss function for all values in `targets` and
            `outputs`.
   :rtype: Number

.. code-block:: jlcon

   julia> value(L1DistLoss(), [1,2,3], [2,5,-2], AvgMode.Sum())
   9

   julia> value(L1DistLoss(), [1.,2,3], [2,5,-2], AvgMode.Sum())
   9.0

   julia> value(L1DistLoss(), [1,2,3], [2,5,-2], AvgMode.Mean())
   3.0

   julia> value(L1DistLoss(), Float32[1,2,3], Float32[2,5,-2], AvgMode.Mean())
   3.0f0

The exact same method signature is also implemented for
:func:`deriv` and :func:`deriv2` respectively.

.. function:: deriv(loss, targets, outputs, avgmode) -> Number

   Computes the **unweighted** sum or mean (depending on
   `avgmode`) of the individual derivatives of the loss function
   for each index-pair in `targets` and `outputs`. This method
   will not allocate a temporary array.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()` or
                               :func:`AvgMode.Mean()`
   :return: The unweighted sum or mean of the element-wise
            derivatives of the loss function for all values in
            `targets` and `outputs`.
   :rtype: Number

.. code-block:: jlcon

   julia> deriv(L2DistLoss(), [1,2,3], [2,5,-2], AvgMode.Sum())
   -2

   julia> deriv(L2DistLoss(), [1,2,3], [2,5,-2], AvgMode.Mean())
   -0.6666666666666665


.. function:: deriv2(loss, targets, outputs, avgmode) -> Number

   Computes the **unweighted** sum or mean (depending on
   `avgmode`) of the individual 2nd derivatives of the loss
   function for each index-pair in `targets` and `outputs`. This
   method will not allocate a temporary array.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()` or
                               :func:`AvgMode.Mean()`
   :return: The unweighted sum or mean of the element-wise
            2nd derivatives of the loss function for all values
            in `targets` and `outputs`.
   :rtype: Number

.. code-block:: jlcon

   julia> deriv2(LogitDistLoss(), [1.,2,3], [2,5,-2], AvgMode.Sum())
   0.49687329928636825

   julia> deriv2(LogitDistLoss(), [1.,2,3], [2,5,-2], AvgMode.Mean())
   0.1656244330954561

Sum and Mean per Observation
-----------------------------

When the targets and predicted outputs are multi-dimensional
arrays instead of vectors, we may be interested in accumulating
the values over all but one dimension. This is typically the case
when we work in a multi-variable regression setting, where each
observation has multiple outputs and thus multiple targets. In
those scenarios we may be more interested in the average loss for
each observation, rather than the total average over all the
data.

To be able to accumulate the values for each observation
separately, we have to explicitly specify the dimension that
denotes the observations. For that purpose we provide the types
contained in the namespace ``ObsDim``.

.. function:: value(loss, targets, outputs, avgmode, obsdim) -> Vector

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually and returns either the
   **unweighted** sum or mean for each observation (depending on
   `avgmode`). This method will not allocate a temporary array,
   but it will allocate the resulting vector.

   Both arrays have to be of the same shape and size. Furthermore
   they have to have at least two array dimensions (i.e. so they
   must not be vectors).

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The multi-dimensional array of
                                 ground truths :math:`\mathbf{y}`.
   :param AbstractArray outputs: The multi-dimensional array of
                                 predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()` or
                               :func:`AvgMode.Mean()`
   :param ObsDimension obsdim: Denotes which of the array
                               dimensions denotes the observations.
                               see ``?ObsDim`` for more information.
   :return: A vector that contains the unweighted sums / means
            of the loss for each observation in `targets` and
            `outputs`.
   :rtype: Vector

Consider the following two matrices, ``targets`` and ``outputs``.
There are two ways to interpret the shape of these arrays if one
dimension is to denote the observations.

.. code-block:: jlcon

   julia> targets = rand(2,4)
   2×4 Array{Float64,2}:
    0.0743675  0.285303  0.247157  0.223666
    0.513145   0.59224   0.32325   0.989964

   julia> outputs = rand(2,4)
   2×4 Array{Float64,2}:
    0.6335    0.319131  0.637087  0.613777
    0.513495  0.264587  0.533555  0.714688

The first interpretation would be to say that the first dimension
denotes the observations. Thus this data would consist of two
observations with four variables each.

.. code-block:: jlcon

   julia> value(L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.First())
   2-element Array{Float64,1}:
    1.373
    0.813583

   julia> value(L1DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.First())
   2-element Array{Float64,1}:
    0.34325
    0.203396

The second possible interpretation would be to say that the
second/last dimension denotes the observations. In that case our
data consists of four observations with two variables each.

.. code-block:: jlcon

   julia> value(L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())
   4-element Array{Float64,1}:
    0.559482
    0.36148
    0.600235
    0.665386

   julia> value(L1DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.Last())
   4-element Array{Float64,1}:
    0.279741
    0.18074
    0.300118
    0.332693

Because this method returns a vector of values, we also provide a
mutating version that can make use a preallocated vector to write
the results into.

.. function:: value!(buffer, loss, targets, outputs, avgmode, obsdim) -> Vector

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually, and computes the the
   **unweighted** sum or mean for each observation, depending on
   `avgmode`. The results are stored into the given vector
   `buffer`. This method will not allocate a temporary array.

   Both arrays have to be of the same shape and size. Furthermore
   they have to have at least two array dimensions (i.e. so they
   must not be vectors).

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param buffer: Array to store the computed values in.
                  Old values will be overwritten and lost.
   :type buffer: `AbstractVector`
   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The multi-dimensional array of
                                 ground truths :math:`\mathbf{y}`.
   :param AbstractArray outputs: The multi-dimensional array of
                                 predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()` or
                               :func:`AvgMode.Mean()`
   :param ObsDimension obsdim: Denotes which of the array
                               dimensions denotes the observations.
                               see ``?ObsDim`` for more information.
   :return: `buffer` (for convenience).

.. code-block:: jlcon

   julia> buffer = zeros(2);

   julia> value!(buffer, L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.First())
   2-element Array{Float64,1}:
    1.373
    0.813583

   julia> value!(buffer, L1DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.First())
   2-element Array{Float64,1}:
    0.34325
    0.203396

   julia> buffer = zeros(4);

   julia> value!(buffer, L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())
   4-element Array{Float64,1}:
    0.559482
    0.36148
    0.600235
    0.665386

   julia> value!(buffer, L1DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.Last())
   4-element Array{Float64,1}:
    0.279741
    0.18074
    0.300118
    0.332693

We also provide both of these methods for :func:`deriv` and
:func:`deriv2` respectively.

.. function:: deriv(loss, targets, outputs, avgmode, obsdim) -> Vector

   same as below but using the 1st derivative.

.. function:: deriv2(loss, targets, outputs, avgmode, obsdim) -> Vector

   Computes the (2nd) derivatives of the loss function for each
   index-pair in `targets` and `outputs` individually and returns
   either the **unweighted** sum or mean for each observation
   (depending on `avgmode`). This method will not allocate a
   temporary array, but it will allocate the resulting vector.

   Both arrays have to be of the same shape and size. Furthermore
   they have to have at least two array dimensions (i.e. so they
   must not be vectors).

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The multi-dimensional array of
                                 ground truths :math:`\mathbf{y}`.
   :param AbstractArray outputs: The multi-dimensional array of
                                 predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()`
                               or :func:`AvgMode.Mean()`
   :param ObsDimension obsdim: Denotes which of the array
                               dimensions denotes the observations.
                               see ``?ObsDim`` for more information.
   :return: A vector that contains the unweighted sums / means
            of the (2nd) loss-derivatives for each observation in
            `targets` and `outputs`.
   :rtype: Vector

.. code-block:: jlcon

   julia> targets = rand(2,4)
   2×4 Array{Float64,2}:
    0.0743675  0.285303  0.247157  0.223666
    0.513145   0.59224   0.32325   0.989964

   julia> outputs = rand(2,4)
   2×4 Array{Float64,2}:
    0.6335    0.319131  0.637087  0.613777
    0.513495  0.264587  0.533555  0.714688

   julia> deriv(L2DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.First())
   2-element Array{Float64,1}:
     2.746
    -0.784548

   julia> deriv(L2DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.First())
   2-element Array{Float64,1}:
     0.686501
    -0.196137

   julia> deriv(L2DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())
   4-element Array{Float64,1}:
     1.11896
    -0.58765
     1.20047
     0.22967

   julia> deriv(L2DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.Last())
   4-element Array{Float64,1}:
     0.559482
    -0.293825
     0.600235
     0.114835

Because this method returns a vector of values, we also provide a
mutating version that can make use a preallocated vector to write
the results into.

.. function:: deriv!(buffer, loss, targets, outputs, avgmode, obsdim) -> Vector

   same as below but using the 1st derivative.

.. function:: deriv2!(buffer, loss, targets, outputs, avgmode, obsdim) -> Vector

   Computes the (2nd) derivatives of the loss function for each
   index-pair in `targets` and `outputs` individually, and
   computes the the unweighted sum or unweighted mean for each
   observation, depending on `avgmode`. The results are stored
   into the given vector `buffer`. This method will not allocate
   a temporary array.

   Both arrays have to be of the same shape and size. Furthermore
   they have to have at least two array dimensions (i.e. so they
   must not be vectors).

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param buffer: Array to store the computed values in.
                  Old values will be overwritten and lost.
   :type buffer: `AbstractVector`
   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The multi-dimensional array of
                                 ground truths :math:`\mathbf{y}`.
   :param AbstractArray outputs: The multi-dimensional array of
                                 predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode avgmode: Must either be :func:`AvgMode.Sum()` or
                               :func:`AvgMode.Mean()`
   :param ObsDimension obsdim: Denotes which of the array
                               dimensions denotes the observations.
                               see ``?ObsDim`` for more information.
   :return: `buffer` (for convenience).

.. code-block:: jlcon

   julia> buffer = zeros(2);

   julia> deriv!(buffer, L2DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.First())
   2-element Array{Float64,1}:
     2.746
    -0.784548

   julia> deriv!(buffer, L2DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.First())
   2-element Array{Float64,1}:
     0.686501
    -0.196137

   julia> buffer = zeros(4);

   julia> deriv!(buffer, L2DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())
   4-element Array{Float64,1}:
     1.11896
    -0.58765
     1.20047
     0.22967

   julia> deriv!(buffer, L2DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.Last())
   4-element Array{Float64,1}:
     0.559482
    -0.293825
     0.600235
     0.114835


Weighted Sum and Mean
-------------------------

Up to now all the averaging was performed in an unweighted
manner. That means that each observation was treated as equal and
had thus the same potential influence on the result. In this
subsection we will consider the situations in which we do want to
explicitly specify the influence of every observation (i.e.
weight them). When we say we weight an observation what it
effectively boils down to is multiplying the result of that
observation (i.e. the computed loss or derivative) with some
number. This is done for every observation.

Let us consider performing a weighting scheme manually. The
following code will compute the loss for three observations, and
multiply the result of the second observation with the number
``2``. If we then sum the results we will see that effectively
the loss of the second observation was counted twice.

.. code-block:: jlcon

   julia> result = value.(L1DistLoss(), [1.,2,3], [2,5,-2]) .* [1,2,1]
   3-element Array{Float64,1}:
    1.0
    6.0
    5.0

   julia> sum(result)
   12.0

The point of weighing observations is to tell the learning
algorithm we are working with, that it is more important to us to
get some observations right than it is for others. So really the
concrete weigh-factor matters less than the ratio between the
different weights. In the example above the second observation
was twice as important any of the other two observations.

In the case of multi-dimensional arrays the process isn't that
simple anymore. In such a scenario, computing the weighted sum
(or weighted mean) can be thought of as having an additional
step. First we either compute the sum or (unweighted!) average
for each observation (which results in a vector), and then we
compute the weighted sum of all observations. The following code
shows how to compute the ``AvgMode.WeightedSum([2,1])`` manually.
This is **not** meant as an example of how to do it, but simply
to show what is happening qualitatively.

.. code-block:: jlcon

   julia> targets = rand(2,4)
   2×4 Array{Float64,2}:
    0.0743675  0.285303  0.247157  0.223666
    0.513145   0.59224   0.32325   0.989964

   julia> outputs = rand(2,4)
   2×4 Array{Float64,2}:
    0.6335    0.319131  0.637087  0.613777
    0.513495  0.264587  0.533555  0.714688

   # WARNING: BAD CODE - ONLY FOR ILLUSTRATION

   julia> tmp = sum(value.(L1DistLoss(), targets, outputs),2) # ObsDim.First()
   2×1 Array{Float64,2}:
    1.373
    0.813584

   julia> sum(x * w for (x,w) in zip(tmp, [2, 1])) # weigh 1st observation twice as high
   3.559587

To manually compute the result for
``AvgMode.WeightedMean([2,1])`` we follow a similar approach, but
use the normalized weight vector in the last step.

.. code-block:: jlcon

   # WARNING: BAD CODE - ONLY FOR ILLUSTRATION

   julia> tmp = mean(value.(L1DistLoss(), targets, outputs),2) # ObsDim.First()
   2×1 Array{Float64,2}:
    0.34325
    0.203396

   julia> sum(x * w for (x,w) in zip(tmp, [0.6666, 0.3333])) # weigh 1st observation twice as high
   0.29660258677499995

Note that you can specify if you want to normalize the weight
vector for both, weighted sum and weighted mean, explicitly. See
the documentation for :class:`AvgMode.WeightedSum` and
:class:`AvgMode.WeightedMean` for more information.

To be able to accumulate the values for each observation
separately when working with multi-dimensional arrays, we have to
explicitly specify the dimension that denotes the observations.
For that purpose we provide the types contained in the namespace
``ObsDim``.

.. function:: value(loss, targets, outputs, weighmode, [obsdim]) -> Number

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually and returns either the
   **weighted** sum or mean for each observation (depending on
   `weighmode`). This method will not allocate a temporary array.
   Both arrays have to be of the same shape and size.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param AverageMode weighmode: Must either be of type
                                 :class:`AvgMode.WeightedSum` or
                                 :class:`AvgMode.WeightedMean`.
                                 Either way, the specified weight
                                 vector must have the same number
                                 of observations as `targets` and
                                 `outputs`.
   :param ObsDimension obsdim: Optional for vectors. Denotes which
                               of the array dimensions denotes the
                               observations.
                               see ``?ObsDim`` for more information.
   :return: A vector that contains the unweighted sums / means
            of the loss for each observation in `targets` and
            `outputs`.
   :rtype: Number



