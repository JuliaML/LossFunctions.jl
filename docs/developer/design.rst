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

Deviations from Literature
----------------------------

Writing Tests
----------------

