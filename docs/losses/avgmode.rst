Efficient Sum and Mean
=========================

In many cases we may not actually be interested in the individual
loss values of the observations, but the sum or mean of them; be
it weighted or unweighted. The naive way to do that is to call
mean on the result of the element-wise operation.

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
accordingly. Bottom line, we allocate temporary memory that we
don't need and could avoid.

For that purpose we provide special methods that compute the
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
the namespace ``AvgMode``.

.. class:: AvgMode.None

   Used by default. This will cause the element-wise results to
   be returned.

.. class:: AvgMode.Sum

   Causes the method to return the unweighted sum of the
   elements instead of the individual elements. Can be used in
   combination with ``ObsDim`` in which case a vector will be
   returned containing the sum for each observation (useful
   mainly for multivariable regression).

.. class:: AvgMode.Mean

   Causes the method to return the unweighted mean of the
   elements instead of the individual elements. Can be used in
   combination with ``ObsDim`` in which case a vector will be
   returned containing the mean for each observation (useful
   mainly for multivariable regression).

.. class:: AvgMode.WeightedSum

   Causes the method to return the weighted sum of all
   observations. The variable ``weights`` has to be a vector of
   the same length as the number of observations. If ``normalize
   = true``, the values of the weight vector will be normalized
   in such as way that they sum to one, instead of used as it.

   .. attribute:: weights

      Vector of weight values that can be used to give certain
      observations a bigger influence on the sum.

      .. code-block:: jlcon

         julia> AvgMode.WeightedSum([1,1,2]);

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

         julia> AvgMode.WeightedMean([1,1,2]);

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

As hinted before, be provide special methods that a implement
memory efficient version of sum and mean for the element-wise
results of :func:`value`. These methods will avoid the allocation
of a temporary array and instead compute the result directly.

When we say "weighted" or "unweighted" accumulation, we are
referring to explicitly specifying the influence of individual
observations on the result by multiplying its value with some
number (i.e. the "weight" of that observation). This implies that
in order to weigh an observation we have to know which array
dimension (if there are more than one) denotes the observations.

For computing an unweighted result, however, we don't need to
know anything about the meaning of the array dimensions as long
as the ``targets`` and the ``outputs`` are of compatible shape
and size.

.. function:: value(loss, targets, outputs, AvgMode.Sum)

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually and returns the
   **unweighted sum** of all values as a `Number`, instead of all
   the individual values as an `Array`. This method will not
   allocate a temporary array.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we want to compute the sum of
                the values for.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :return: The unweighted sum of the element-wise values of the
            loss function for all values in `targets` and `outputs`.
   :rtype: `Number`

.. code-block:: jlcon

   julia> value(L1DistLoss(), [1,2,3], [2,5,-2], AvgMode.Sum())
   9

   julia> value(L1DistLoss(), [1.,2,3], [2,5,-2], AvgMode.Sum())
   9.0

.. function:: value(loss, targets, outputs, AvgMode.Mean)

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually and returns the
   **unweighted mean** of all values as a `Number`, instead of
   all the individual values as as `Array`. This method will not
   allocate a temporary array.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we want to compute the mean of
                the values for.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :return: The unweighted mean of the element-wise values of the
            loss function for all values in `targets` and
            `outputs`.
   :rtype: `Number`

.. code-block:: jlcon

   julia> value(L1DistLoss(), [1,2,3], [2,5,-2], AvgMode.Mean())
   3.0

   julia> value(L1DistLoss(), Float32[1,2,3], Float32[2,5,-2], AvgMode.Mean())
   3.0f0


Sum and Mean per Observation
-----------------------------

In some of the situations, in which the targets and predicted
outputs are multi-dimensional arrays instead of vectors, we may
be interested in accumulating the values over all but one
dimension. This is usually the case when we work in a
multi-variable regression setting, where each observation has
multiple outputs and thus multiple targets. In those scenarios we
may be interested in the average loss for each observation rather
than all the individual values or the total average over all the
data.

To be able to accumulate the values for each observation
separately, we have to explicitly specify the dimension that
denotes the observations. For that purpose we provide the types
contained in the namespace ``ObsDim``.

.. function:: value(loss, targets, outputs, AvgMode.Sum, obsdim)

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually and returns the
   **unweighted sum** for each observation separately (i.e. as a
   vector). This method will not allocate a temporary array, but
   it will allocate the resulting vector.

   Both arrays have to be of the same shape and size. Furthermore
   they have to have at least two array dimensions (i.e. so not
   vectors).

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The multi-dimensional array of
                                 ground truths :math:`\mathbf{y}`.
   :param AbstractArray outputs: The multi-dimensional array of
                                 predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :param ObsDimension obsdim: Denotes which of the array
                               dimensions denotes the observations.
                               see ``?ObsDim`` for more information.
   :return: A vector that contains the unweighted sums for each
            observation in `targets` and `outputs`.
   :rtype: `Vector`

.. code-block:: jlcon

   julia> targets = rand(2,4)
   2×4 Array{Float64,2}:
    0.0743675  0.285303  0.247157  0.223666
    0.513145   0.59224   0.32325   0.989964

   julia> outputs = rand(2,4)
   2×4 Array{Float64,2}:
    0.6335    0.319131  0.637087  0.613777
    0.513495  0.264587  0.533555  0.714688

   julia> value(L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.First())
   2-element Array{Float64,1}:
    1.373
    0.813583

   julia> value(L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())
   4-element Array{Float64,1}:
    0.559482
    0.36148
    0.600235
    0.665386

Since we need to return a vector of values, we also provide a
mutating version that can use a preallocated vector to write the
results into.

.. function:: value!(buffer, loss, targets, outputs, AvgMode.Sum, obsdim)

   Computes the values of the loss function for each index-pair
   in `targets` and `outputs` individually, computes the
   **unweighted sum** for each observation separately, and then
   store them into the given vector `buffer`. This method will
   not allocate a temporary array.

   Both arrays have to be of the same shape and size. Furthermore
   they have to have at least two array dimensions (i.e. so not
   vectors).

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

   julia> buffer = zeros(4);

   julia> value!(buffer, L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())
   4-element Array{Float64,1}:
    0.559482
    0.36148
    0.600235
    0.665386

