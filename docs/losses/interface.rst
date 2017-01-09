Working with Losses
=====================

Even though they are called loss "functions", this package
implements them as immutable types instead of true Julia
functions. There are good reasons for that. For example it allows
us to specify the properties of losses explicitly (e.g.
``isconvex(myloss)``). It also makes for a more consistent API
when it comes to computing the value or the derivative. Some loss
functions even have additional parameters that need to be
specified, such as the :math:`\epsilon` in the case of the
:math:`\epsilon`-insensitive loss. In order to avoid potential
confusions with true Julia functions, we will refer to "loss
functions" as "losses" instead.

The available losses share a common interface for the most part.
This section will provide an overview of the basic functionality
that is available for all the different types of losses. We will
discuss how to create a loss, how to compute its value and
derivative, and how to query its properties.

Instantiating a Loss
-----------------------

Losses are immutable types. As such, one has to instantiate one
in order to work with it. For most losses, the constructors do
not expect any parameters.

.. code-block:: jlcon

   julia> L2DistLoss()
   LossFunctions.LPDistLoss{2}()

   julia> HingeLoss()
   LossFunctions.L1HingeLoss()

We just said that we need to instantiate a loss in order to work
with it. One could be inclined to belief, that it would be more
memory-efficient to "pre-allocate" a loss when using it in more
than one place.

.. code-block:: jlcon

   julia> loss = L2DistLoss()
   LossFunctions.LPDistLoss{2}()

   julia> value(loss, 2, 3)
   1

However, that is a common oversimplification. Because all losses
are immutable types, they can live on the stack and thus do not
come with a heap-allocation overhead.

Even more interesting, for such losses as :class:`L2DistLoss`,
which don not have any constructor parameters or member variables,
there is no additional code executed at all. Such singletons are
only used for dispatch and don't even produce any additional
code, which you can observe for yourself in the code below. As
such they are zero-cost abstractions.

.. code-block:: jlcon

   julia> v1(loss,t,y) = value(loss,t,y)

   julia> v2(t,y) = value(L2DistLoss(),t,y)

   julia> @code_llvm v1(loss, 2, 3)
   define i64 @julia_v1_70944(i64, i64) #0 {
   top:
     %2 = sub i64 %1, %0
     %3 = mul i64 %2, %2
     ret i64 %3
   }

   julia> @code_llvm v2(2, 3)
   define i64 @julia_v2_70949(i64, i64) #0 {
   top:
     %2 = sub i64 %1, %0
     %3 = mul i64 %2, %2
     ret i64 %3
   }

On the other hand, some types of losses are actually more
comparable to whole families of losses instead of a single one.
For example, the immutable type :class:`L1EpsilonInsLoss` has a
free parameter :math:`\epsilon`. Each concrete :math:`\epsilon`
results in a different concrete loss of the same family of
epsilon-insensitive losses.

.. code-block:: jlcon

   julia> L1EpsilonInsLoss(0.5)
   LossFunctions.L1EpsilonInsLoss{Float64}(0.5)

   julia> L1EpsilonInsLoss(1)
   LossFunctions.L1EpsilonInsLoss{Float64}(1.0)

For such losses that do have parameters, it can make a slight
difference to pre-instantiate a loss. While they will live on the
stack, the constructor usually performs some assertions and
conversion for the given parameter. This can come at a slight
overhead. At the very least it will not produce the same exact
code when pre-instantiated. Still, the fact that they are immutable
makes them very efficient abstractions with little to no
performance overhead, and zero memory allocations on the heap.

Computing the Values
-----------------------

The first thing we may want to do is compute the loss for some
observation (singular). In fact, all losses are implemented on
single values under the hood. The core function to compute the
value of a loss is :func:`value`. We will see throughout the
documentation that it allows for a lot of different method
signatures to accomplish a variety of tasks.

.. function:: value(loss, target, output)

   Computes the result for the loss-function denoted by the
   parameter `loss`. Note that `target` and `output` can be of
   different numeric type, in which case promotion is performed
   in the manner appropriate for the given loss.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   .. math:: L : Y \times \mathbb{R} \rightarrow [0,\infty)

   :param loss: The loss-function :math:`L` we want to compute the
                value with.
   :type loss: :class:`SupervisedLoss`
   :param Number target: The ground truth :math:`y \in Y` of the
                         observation.
   :param Number output: The predicted output :math:`\hat{y} \in
                         \mathbb{R}` for the observation.
   :return: The (non-negative) numeric result of the loss-function
            for the given parameters.
   :rtype: `Number`

.. code-block:: jlcon

   #               loss        y    ŷ
   julia> value(L1DistLoss(), 1.0, 2.0)
   1.0

   julia> value(L1DistLoss(), 1, 2)
   1

   julia> value(L1HingeLoss(), -1, 2)
   3

   julia> value(L1HingeLoss(), -1f0, 2f0)
   3.0f0

It may be interesting to note, that this function also supports
broadcasting and all the syntax benefits that come with it. Thus,
it is quite simple to make use of preallocated memory for storing
the element-wise results.

.. code-block:: jlcon

   julia> value.(L1DistLoss(), [1,2,3], [2,5,-2])
   3-element Array{Int64,1}:
    1
    3
    5

   julia> buffer = zeros(3); # preallocate a buffer

   julia> buffer .= value.(L1DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
    1.0
    3.0
    5.0

Furthermore, with the loop fusion changes that were introduced in
Julia 0.6, one can also easily weight the influence of each
observation without allocating a temporary array.

.. code-block:: jlcon

   julia> buffer .= value.(L1DistLoss(), [1.,2,3], [2,5,-2]) .* [2,1,0.5]
   3-element Array{Float64,1}:
    2.0
    3.0
    2.5

Even though broadcasting is supported, we do expose a vectorized
method natively. This is done mainly for API consistency reasons.
Internally it even uses broadcast itself, but it does provide the
additional benefit of a more reliable type-inference.

.. function:: value(loss, targets, outputs)

   Computes the value of the loss function for each index-pair
   in `targets` and `outputs` individually and returns the result
   as an array of the appropriate size.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we want to compute the values for.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :return: The element-wise results of the loss function for all
            values in `targets` and `outputs`.
   :rtype: `AbstractArray`

.. code-block:: jlcon

   julia> value(L1DistLoss(), [1,2,3], [2,5,-2])
   3-element Array{Int64,1}:
    1
    3
    5

   julia> value(L1DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
    1.0
    3.0
    5.0

We also provide a mutating version for the same reasons. It
even utilizes ``broadcast!`` underneath.

.. function:: value!(buffer, loss, targets, outputs)

   Computes the value of the loss function for each index-pair in
   `targets` and `outputs` individually, and stores them in the
   preallocated `buffer`, which has to be of the appropriate
   size.

   In the case that the two parameters `targets` and `outputs`
   are arrays with a different number of dimensions, broadcast
   will be performed. Note that the given parameters are
   expected to have the same size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param buffer: Array to store the computed values in.
                  Old values will be overwritten and lost.
   :type buffer: `AbstractArray`
   :param loss: The loss-function we want to compute the values for.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :return: `buffer` (for convenience).

.. code-block:: jlcon

   julia> buffer = zeros(3); # preallocate a buffer

   julia> value!(buffer, L1DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
    1.0
    3.0
    5.0



Computing the Derivatives
---------------------------

Maybe the more interesting aspect of loss functions are their
derivatives. In fact, most of the popular learning algorithm in
ML, such as gradient descent, utilize the derivatives of the loss
in one way or the other during the training process.

To compute the derivative of some loss we expose the function
:func:`deriv`. It supports the same exact method signatures as
:func:`value`. Note that we always compute the derivative in
respect to the predicted output, since we are interested in which
direction the output should change.

.. function:: deriv(loss, target, output)

   Computes the derivative for the loss-function denoted by the
   parameter `loss` in respect to the `output`. Note that
   `target` and `output` can be of different numeric type, in
   which case promotion is performed in the manner appropriate
   for the given loss.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function :math:`L` we want to compute the
                derivative with.
   :type loss: :class:`SupervisedLoss`
   :param Number target: The ground truth :math:`y \in Y` of the
                         observation.
   :param Number output: The predicted output :math:`\hat{y} \in
                         \mathbb{R}` for the observation.
   :return: The derivative of the loss-function for the given
            parameters.
   :rtype: `Number`

.. code-block:: jlcon

   #               loss        y    ŷ
   julia> deriv(L2DistLoss(), 1.0, 2.0)
   2.0

   julia> deriv(L2DistLoss(), 1, 2)
   2

   julia> deriv(L2HingeLoss(), -1, 2)
   6

   julia> deriv(L2HingeLoss(), -1f0, 2f0)
   6.0f0

Similar to :func:`value`, this function also supports
broadcasting and all the syntax benefits that come with it. Thus,
one can make use of preallocated memory for storing the
element-wise derivatives.

.. code-block:: jlcon

   julia> deriv.(L2DistLoss(), [1,2,3], [2,5,-2])
   3-element Array{Int64,1}:
      2
      6
    -10

   julia> buffer = zeros(3); # preallocate a buffer

   julia> buffer .= deriv.(L2DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
      2.0
      6.0
    -10.0

Furthermore, with the loop fusion changes that were introduced in
Julia 0.6, one can also easily weight the influence of each
observation without allocating a temporary array.

.. code-block:: jlcon

   julia> buffer .= deriv.(L2DistLoss(), [1.,2,3], [2,5,-2]) .* [2,1,0.5]
   3-element Array{Float64,1}:
     4.0
     6.0
    -5.0

We do expose a vectorized method natively. This is done mainly
for API consistency reasons. Internally it even uses broadcast
itself, but it does provide the additional benefit of a more
reliable type-inference.

.. function:: deriv(loss, targets, outputs)

   Computes the derivative of the loss function in respect to the
   output for each index-pair in `targets` and `outputs`
   individually and returns the result as an array of the
   appropriate size.

   In the case that the two parameters are arrays with a
   different number of dimensions, broadcast will be performed.
   Note that the given parameters are expected to have the same
   size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we want to compute the
                derivative for.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :return: The element-wise derivatives of the loss function for
            all values in `targets` and `outputs`.
   :rtype: `AbstractArray`

.. code-block:: jlcon

   julia> deriv(L2DistLoss(), [1,2,3], [2,5,-2])
   3-element Array{Int64,1}:
      2
      6
    -10

   julia> deriv(L2DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
      2.0
      6.0
    -10.0

We also provide a mutating version for the same reasons. It
even utilizes ``broadcast!`` underneath.

.. function:: deriv!(buffer, loss, targets, outputs)

   Computes the derivatives of the loss function in respect to
   the outputs for each index-pair in `targets` and `outputs`
   individually, and stores them in the preallocated `buffer`,
   which has to be of the appropriate size.

   In the case that the two parameters `targets` and `outputs`
   are arrays with a different number of dimensions, broadcast
   will be performed. Note that the given parameters are
   expected to have the same size in the dimensions they share.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param buffer: Array to store the computed derivatives in.
                  Old values will be overwritten and lost.
   :type buffer: `AbstractArray`
   :param loss: The loss-function we want to compute the
                derivatives for.
   :type loss: :class:`SupervisedLoss`
   :param AbstractArray targets: The array of ground truths
                                 :math:`\mathbf{y}`.
   :param AbstractArray outputs: The array of predicted outputs
                                 :math:`\mathbf{\hat{y}}`.
   :return: `buffer` (for convenience).

.. code-block:: jlcon

   julia> buffer = zeros(3); # preallocate a buffer

   julia> deriv!(buffer, L2DistLoss(), [1.,2,3], [2,5,-2])
   3-element Array{Float64,1}:
      2.0
      6.0
    -10.0

It is also possible to compute the value and derivative at the
same time. For some losses that means less computation overhead.

.. function:: value_deriv(loss, target, output)

   Returns the results of :func:`value` and :func:`deriv` as a
   tuple. In some cases this function can yield better
   performance, because the losses can make use of shared
   variable when computing the values. Note that `target` and
   `output` can be of different numeric type, in which case
   promotion is performed in the manner appropriate for the given
   loss.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function we are working with.
   :type loss: :class:`SupervisedLoss`
   :param Number target: The ground truth :math:`y \in Y` of the
                         observation.
   :param Number output: The predicted output :math:`\hat{y} \in
                         \mathbb{R}` for the observation.
   :return: The value and the derivative of the loss-function for
            the given parameters.
   :rtype: `Tuple`

.. code-block:: jlcon

   #                     loss         y    ŷ
   julia> value_deriv(L2DistLoss(), -1.0, 3.0)
   (16.0,8.0)

Closures
-------------

In some circumstances it may be convenient to have the loss function
or its derivative as a proper Julia function. Closures provide
just that as the following examples demonstrate.

.. code-block:: julia

   f = value_fun(L2DistLoss())
   f(targets, outputs) # computes the value of L2DistLoss

   d = deriv_fun(L2DistLoss())
   d(targets, outputs) # computes the derivative of L2DistLoss


.. function:: value_fun(loss)

.. function:: deriv_fun(loss)

.. function:: deriv2_fun(loss)

.. function:: value_deriv_fun(loss)

Querying loss properties
~~~~~~~~~~~~~~~~~~~~~~~~~~

The losses implemented in this package provide a range of properties
that can be queried by functions defined in *LearnBase.jl*.

.. function:: isminimizable(loss)

.. function:: isconvex(loss)

.. function:: isstrictlyconvex(loss)

.. function:: isstronglyconvex(loss)

.. function:: isdifferentiable(loss[, at])

.. function:: istwicedifferentiable(loss[, at])

.. function:: isnemitski(loss)

.. function:: islipschitzcont(loss)

.. function:: islocallylipschitzcont(loss)

.. function:: isclipable(loss)

.. function:: ismarginbased(loss)

.. function:: isclasscalibrated(loss)

.. function:: isdistancebased(loss)

.. function:: issymmetric(loss)



Distance-based Losses
----------------------

.. class:: DistanceLoss

   Abstract subtype of :class:`SupervisedLoss`.
   A supervised loss that can be simplified to
   ``L(targets, outputs) = L(targets - outputs)`` is considered
   distance-based.

.. function:: value(loss, difference)

   Computes the value of the loss function for each
   observation in ``difference`` individually and returns the result
   as an array of the same size as the parameter.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`DistanceLoss`
   :param difference: The result of subtracting the true targets from
                      the predicted outputs.
   :type difference: ``AbstractArray``
   :return: The value of the loss function for the elements in
            ``difference``.
   :rtype: ``AbstractArray``

.. function:: deriv(loss, difference)

   Computes the derivative of the loss function for each
   observation in ``difference`` individually and returns the result
   as an array of the same size as the parameter.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`DistanceLoss`
   :param difference: The result of subtracting the true targets from
                      the predicted outputs.
   :type difference: ``AbstractArray``
   :return: The derivatives of the loss function for the elements in
            ``difference``.
   :rtype: ``AbstractArray``

.. function:: value_deriv(loss, difference)

   Returns the results of :func:`value` and :func:`deriv` as a tuple.
   In some cases this function can yield better performance, because
   the losses can make use of shared variable when computing
   the values.



Margin-based Losses
--------------------

.. class:: MarginLoss

   Abstract subtype of :class:`SupervisedLoss`.
   A supervised loss, where the targets are in {-1, 1}, and which
   can be simplified to ``L(targets, outputs) = L(targets * outputs)``
   is considered margin-based.

.. function:: value(loss, agreement)

   Computes the value of the loss function for each
   observation in ``agreement`` individually and returns the result
   as an array of the same size as the parameter.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`MarginLoss`
   :param agreement: The result of multiplying the true targets with
                     the predicted outputs.
   :type agreement: ``AbstractArray``
   :return: The value of the loss function for the elements in
            ``agreement``.
   :rtype: ``AbstractArray``

.. function:: deriv(loss, agreement)

   Computes the derivative of the loss function for each
   observation in ``agreement`` individually and returns the result
   as an array of the same size as the parameter.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`MarginLoss`
   :param agreement: The result of multiplying the true targets with
                     the predicted outputs.
   :type agreement: ``AbstractArray``
   :return: The derivatives of the loss function for the elements in
            ``agreement``.
   :rtype: ``AbstractArray``

.. function:: value_deriv(loss, agreement)

   Returns the results of :func:`value` and :func:`deriv` as a tuple.
   In some cases this function can yield better performance, because
   the losses can make use of shared variable when computing
   the values.

