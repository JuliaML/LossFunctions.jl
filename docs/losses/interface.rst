Shared Interface
=================

All provided loss functions share a common interface with other
losses, but to a varying degree.

This section will provide an overview of the shared functions are
available to the different sub-class of an abstract ``Loss``

Supervised Losses
------------------

.. class:: SupervisedLoss

   Abstract subtype of ``Loss``.
   A loss is considered **supervised**, if all the information needed
   to compute ``value(loss, features, targets, outputs)`` are contained
   in ``targets`` and ``outputs``, and thus allows for the
   simplification ``value(loss, targets, outputs)``.

Computing the values
~~~~~~~~~~~~~~~~~~~~~

.. function:: value(loss, targets, outputs)

   Computes the value of the loss function for each observation-pair
   in ``targets`` and ``outputs`` individual and returns the result
   as an array of the same size as the parameters.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param targets: The true targets that we would like your model
                   to predict.
   :type targets: ``AbstractArray``
   :param outputs: The predicted outputs that your model produced.
   :type outputs: ``AbstractArray``
   :return: The values of the loss function for the elements in
            ``targets`` and ``outputs``.
   :rtype: ``AbstractArray``

.. function:: sumvalue(loss, targets, outputs)

   Same as :func:`value`, but returns the **sum** of all values as
   a ``Number`` instead of all the individual values as ``Array``.

   :rtype: ``Number``

.. function:: meanvalue(loss, targets, outputs)

   Same as :func:`value`, but returns the unweighted **mean** of all
   values as a single ``Number`` instead of all the individual values
   as ``Array``.

   :rtype: ``Number``

.. function:: value!(buffer, loss, targets, outputs)

   Computes the values of the loss function for each observation-pair
   in ``targets`` and ``outputs`` individually and stores them in
   the preallocated ``buffer``, which has to be the same size as
   the parameters.

   :param buffer: Array to store the computed values in.
                  Old values will be overwritten and lost.
   :type buffer: ``AbstractArray``
   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param targets: The true targets that we would like your model
                   to predict.
   :type targets: ``AbstractArray``
   :param outputs: The predicted outputs that your model produced.
   :type outputs: ``AbstractArray``
   :return: ``buffer``

Computing the derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. function:: deriv(loss, targets, outputs)

   Computes the derivative of the loss function for each
   observation-pair in ``targets`` and ``outputs`` individually and
   returns the result as an array of the same size as the parameters.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param targets: The true targets that we would like your model
                   to predict.
   :type targets: ``AbstractArray``
   :param outputs: The predicted outputs that your model produced.
   :type outputs: ``AbstractArray``
   :return: The derivatives of the loss function for the elements in
            ``targets`` and ``outputs``.
   :rtype: ``AbstractArray``

.. function:: sumderiv(loss, targets, outputs)

   Same as :func:`deriv`, but returns the **sum** of all derivatives
   as a single ``Number``, instead of all the individual derivatives
   as ``Array``.

   :rtype: ``Number``

.. function:: meanderiv(loss, targets, outputs)

   Same as :func:`deriv`, but returns the unweighted **mean** of all
   derivatives as a single ``Number``, instead of all the individual
   derivatives as ``Array``.

   :rtype: ``Number``

.. function:: deriv!(buffer, loss, targets, outputs)

   Computes the derivative of the loss function for each
   observation-pair in ``targets`` and ``outputs`` individually and
   stores them in the preallocated ``buffer``, which has to be the
   same size as the parameters.

   :param buffer: Array to store the computed derivatives in.
                  Old values will be overwritten and lost.
   :type buffer: ``AbstractArray``
   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`SupervisedLoss`
   :param targets: The true targets that we would like your model
                   to predict.
   :type targets: ``AbstractArray``
   :param outputs: The predicted outputs that your model produced.
   :type outputs: ``AbstractArray``
   :return: ``buffer``

.. function:: value_deriv(loss, targets, outputs)

   Returns the results of :func:`value` and :func:`deriv` as a tuple.
   In some cases this function can yield better performance, because
   the losses can make use of shared variable when computing
   the values.

Closures
~~~~~~~~~~

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
   observation in ``difference` individually and returns the result
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
   observation in ``agreement` individually and returns the result
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

