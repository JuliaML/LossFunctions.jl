Developer Documentation
=========================

In this part of the documentation we will discuss some of the
internal design aspects of this library. Consequently, the target
audience of this section and its sub-sections is primarily people
interested in contributing to this package. As such, the
information provided here should be of little to no relevance for
users interested in simply applying the package.

Abstract Superclasses
--------------------------

We have seen in previous sections, that many families of loss
functions are implemented as immutable types with free
parameters. An example for such a family is the
:class:`L1EpsilonInsLoss`, which represents all the
:math:`\epsilon`-insensitive loss-functions for each possible
value of :math:`\epsilon`.

Aside from these special families, there a handful of more
generic families that between them contain almost all of the loss
functions this package implements. These families are defined as
abstract types in the type tree. Their main purpose is two-fold:

- From an end-user's perspective, they are most useful for
  dispatching on the particular kind of prediction problem that
  they are intended for (regression vs classification).

- Form an implementation perspective, these abstract types allow
  us to implement shared functionality and fall-back methods,
  or even allow for a simpler implementation.

Most of the implemented loss functions fall under the umbrella of
supervised losses. As such, we barely mention other types of
losses anywhere in this documentation.

.. class:: SupervisedLoss

   Abstract subtype of :class:`Loss`.

   As mentioned in the background section, a supervised loss is a
   family of functions of two parameters, namely the true targets
   and the predicted outcomes. A loss is considered
   **supervised**, if all the information needed to compute
   ``value(loss, features, target, output)`` are contained in
   ``target`` and ``output``, and thus allows for the
   simplification ``value(loss, target, output)``.

There are two interesting sub-families of supervised loss
functions.  One of these families is called distance-based. All
losses that belong to this family are implemented as subtype of
the abstract type :class:`DistanceLoss`, which itself is subtype
of :class:`SupervisedLoss`.

.. class:: DistanceLoss

   Abstract subtype of :class:`SupervisedLoss`. A supervised loss
   that can be simplified to ``value(loss, target, output)`` =
   ``value(loss, output - target)`` is considered distance-based.

The second core sub-family of supervised losses is called
margin-based. All loss functions that belong to this family are
implemented as subtype of the abstract type :class:`MarginLoss`,
which itself is subtype of :class:`SupervisedLoss`.

.. class:: MarginLoss

   Abstract subtype of :class:`SupervisedLoss`. A supervised
   loss, where the targets are in {-1, 1}, and which can be
   simplified to ``value(loss, target, output)`` = ``value(loss,
   target * output)`` is considered margin-based.

Shared Interface
----------------------

Each of the three abstract types listed above serves a purpose
other than dispatch. All losses that belong to the same family
share functionality to some degree. For example all subtypes of
:class:`SupervisedLoss` share the same implementations for the
vectorized versions of :func:`value` and :func:`deriv`.

More interestingly, the abstract types :class:`DistanceLoss` and
:class:`MarginLoss`, serve an additional purpose aside from
shared functionality. We have seen in the background section what
it is that makes a loss margin-based or distance-based. Without
repeating the definition let us state that it boils down to the
existence of a *representing function* :math:`\psi`, which allows
to compute a loss using a unary function instead of a binary one.
Indeed, all the subtypes of :class:`DistanceLoss` and
:class:`MarginLoss` are implemented in the unary form of their
representing function.

Distance-based Losses
~~~~~~~~~~~~~~~~~~~~~~

Supervised losses that can be expressed as a univariate function
of ``output - target`` are referred to as distance-based losses.
Distance-based losses are typically utilized for regression
problems. That said, there are also other losses that are useful
for regression problems that don't fall into this category, such
as the :class:`PeriodicLoss`.

.. function:: value(loss, difference)

   Computes the value of the representing function :math:`\psi`
   of the given `loss` at `difference`.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`DistanceLoss`
   :param difference: The result of subtracting the true target
                      :math:`y` from the predicted output
                      :math:`\hat{y}`.
   :type difference: `Number`
   :return: The value of the losses representing function at
            the point `difference`.
   :rtype: `Number`

.. function:: deriv(loss, difference)

   Computes the derivative of the representing function
   :math:`\psi` of the given `loss` at `difference`.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`DistanceLoss`
   :param difference: The result of subtracting the true target
                      :math:`y` from the predicted output
                      :math:`\hat{y}`.
   :type difference: `Number`
   :return: The derivativ of the losses representing function at
            the point `difference`.
   :rtype: `Number`

.. function:: value_deriv(loss, difference)

   Returns the results of :func:`value` and :func:`deriv` as a
   tuple. In some cases this function can yield better
   performance, because the losses can make use of shared
   variable when computing the values.

Margin-based Losses
~~~~~~~~~~~~~~~~~~~~~~~~~~

Margin-based losses are supervised losses where the values of the
targets are restricted to be in :math:`\{1,-1\}`, and which can
be expressed as a univariate function ``output * target``.

.. function:: value(loss, agreement)

   Computes the value of the representing function :math:`\psi`
   of the given `loss` at `agreement`.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`MarginLoss`
   :param agreement: The result of multiplying the true target
                     :math:`y` with the predicted output
                     :math:`\hat{y}`.
   :type agreement: `Number`
   :return: The value of the losses representing function
            at the given point `agreement`.
   :rtype: `Number`

.. function:: deriv(loss, agreement)

   Computes the derivative of the representing function
   :math:`\psi` of the given `loss` at `agreement`.

   :param loss: An instance of the loss we are interested in.
   :type loss: :class:`MarginLoss`
   :param agreement: The result of multiplying the true target
                     :math:`y` with the predicted output
                     :math:`\hat{y}`.
   :type agreement: `Number`
   :return: The derivative of the losses representing function
            at the given point `agreement`.
   :rtype: `Number`

.. function:: value_deriv(loss, agreement)

   Returns the results of :func:`value` and :func:`deriv` as a
   tuple. In some cases this function can yield better
   performance, because the losses can make use of shared
   variable when computing the values.

Writing Tests
----------------

.. warning::

   This section is still under development and thus in an
   unfinished state.

