Developer Documentation
=========================

Abstract Superclasses
--------------------------

Most of the implemented losses fall under the category of
supervised losses. In other words they represent functions with
two parameters (the true targets and the predicted outcomes) to
compute their value.

.. class:: SupervisedLoss

   Abstract subtype of ``Loss``.
   A loss is considered **supervised**, if all the information needed
   to compute ``value(loss, features, targets, outputs)`` are contained
   in ``targets`` and ``outputs``, and thus allows for the
   simplification ``value(loss, targets, outputs)``.

.. class:: DistanceLoss

   Abstract subtype of :class:`SupervisedLoss`.
   A supervised loss that can be simplified to
   ``L(targets, outputs) = L(targets - outputs)`` is considered
   distance-based.

.. class:: MarginLoss

   Abstract subtype of :class:`SupervisedLoss`.
   A supervised loss, where the targets are in {-1, 1}, and which
   can be simplified to ``L(targets, outputs) = L(targets * outputs)``
   is considered margin-based.

Shared Interface
-------------------

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

Shared Interface
-------------------

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


Deviations from Literature
----------------------------

Writing Tests
----------------

