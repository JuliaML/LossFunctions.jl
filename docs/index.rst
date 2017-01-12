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

This section gives a more detailed treatment of the exposed
functions and their available methods. We will start by
describing the basic interface that all loss-functions share.

.. toctree::
   :maxdepth: 2

   losses/interface

Next we will consider how to average or sum the results of the
loss-functions more efficiently. Because we are only interested
in the sum or average we can avoid allocating a temporary array.

.. toctree::
   :maxdepth: 2

   losses/avgmode
   losses/other

Provided Lossfunctions
--------------------------------

Aside from the interface, this package also provides a number of
popular (and not so popular) loss functions out-of-the-box. Great
effort has been put into ensuring a correct, efficient, and
type-stable implementation for those. Most of them either belong
to the family of distance-based or margin-based losses. These two
categories are also indicative for if a loss is intended for
regression or classification problems

+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+
| Distance-based Losses (Regression)                                                     | Margin-based Losses (Classification)                                                   |
+========================================================================================+========================================================================================+
| .. image:: https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/distance.svg | .. image:: https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/margin.svg   |
+----------------------------------------------------------------------------------------+----------------------------------------------------------------------------------------+

The loss functions, that belong to the category "distance-based",
are primarily used in regression problems. They utilize the
numeric difference between the true target and the predicted
output is used as a proxy variable to quantify the quality of
individual predictions.

.. toctree::
   :maxdepth: 2

   losses/distance

.. toctree::
   :maxdepth: 2

   losses/margin

Internals
--------------------------------

If you are interested in contributing to LossFunctions.jl, or
simply want to understand how and why the package does then take
a look at our developer documentation.

.. toctree::
   :maxdepth: 2

   developer/design

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

