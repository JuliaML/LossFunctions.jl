LossFunctions.jl's documentation
=================================

This package represents a community effort to centralize the
definition and implementation of **loss functions** in Julia.
As such, it is a part of the `JuliaML <https://github.com/JuliaML>`_
ecosystem.

The sole purpose of this package is to provide an efficient and
extensible implementation of various loss functions used in
Machine Learning. It is thus intended to serve as a special
purpose back-end for other ML libraries that require losses to
provide their functionality. To that end we provide a large list
of implemented loss functions as well as an API to query their
properties such as convexity. Furthermore we expose methods to
compute their values, derivatives, and second derivatives for
single observations as well as arbitrarily sized arrays of
observations. In the case of arrays a user has the ability to
define if and how element-wise results are averaged or summed
over.

From an end-user's perspective one normally does not need to
import this package directly. That said it can provide a decent
starting point for students that are interested in investigating
the properties and behaviour of loss functions.

Where to begin?
---------------------

If this is the first time you consider using LossFunctions for your
machine learning related experiments or packages, make sure to
check out the "Getting Started" section.

.. toctree::
   :maxdepth: 2

   introduction/gettingstarted

Introduction and Motivation
------------------------------

.. toctree::
   :maxdepth: 2

   introduction/motivation

API Documentation
--------------------------------

For details on a specific aspect, see the documentation outlined below.

.. toctree::
   :maxdepth: 2

   losses/interface
   losses/distance
   losses/margin
   losses/other

Indices and tables
==================

.. toctree::
   :hidden:
   :maxdepth: 2

   about/acknowledgements
   about/license

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

