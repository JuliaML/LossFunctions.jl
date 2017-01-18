LossFunctions.jl's documentation
=================================

This package represents a community effort to centralize the
definition and implementation of **loss functions** in Julia.
As such, it is a part of the `JuliaML <https://github.com/JuliaML>`_
ecosystem.

The sole purpose of this package is to provide an efficient and
extensible implementation of various loss functions used
throughout Machine Learning (ML). It is thus intended to serve as
a special purpose back-end for other ML libraries that require
losses to accomplish their tasks. To that end we provide a
considerable amount of carefully implemented loss functions, as
well as an API to query their properties (e.g. convexity).
Furthermore, we expose methods to compute their values,
derivatives, and second derivatives for single observations as
well as arbitrarily sized arrays of observations. In the case of
arrays a user additionally has the ability to define if and how
element-wise results are averaged or summed over.

From an end-user's perspective one normally does not need to
import this package directly. That said, it should provide a
decent starting point for any student that is interested in
investigating the properties or behaviour of loss functions.

Where to begin?
---------------------

If this is the first time you consider using LossFunctions for
your machine learning related experiments or packages, make sure
to check out the "Getting Started" section.

.. toctree::
   :maxdepth: 2

   introduction/gettingstarted

Introduction and Motivation
------------------------------

If you are new to Machine Learning in Julia, or are simply
interested in how and why this package works the way it works,
feel free to take a look at the following sections. There we
discuss the concepts involved and outline the most important
terms and definitions.

.. toctree::
   :maxdepth: 2

   introduction/motivation

API Documentation
--------------------------------

This section gives a more detailed treatment of the exposed
functions and their available methods. We will start by
describing how to instantiate a loss, as well as the basic
interface that all loss functions share.

.. toctree::
   :maxdepth: 2

   losses/interface

Next we will consider how to average or sum the results of the
loss functions more efficiently. The methods described here are
implemented in such a way as to avoid allocating a temporary
array.

.. toctree::
   :maxdepth: 2

   losses/avgmode
   losses/other

Available Loss Functions
--------------------------------

Aside from the interface, this package also provides a number of
popular (and not so popular) loss functions out-of-the-box. Great
effort has been put into ensuring a correct, efficient, and
type-stable implementation for those. Most of them either belong
to the family of distance-based or margin-based losses. These two
categories are also indicative for if a loss is intended for
regression or classification problems

Loss Functions for Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Loss functions that belong to the category "distance-based" are
primarily used in regression problems. They utilize the numeric
difference between the predicted output and the true target as a
proxy variable to quantify the quality of individual predictions.

+----------------------------------------------+----------------------------------------------------------------------------------------+
| .. toctree::                                 | .. image:: https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/distance.svg |
|    :maxdepth: 2                              |                                                                                        |
|                                              |                                                                                        |
|    losses/distance                           |                                                                                        |
+----------------------------------------------+----------------------------------------------------------------------------------------+

Loss Functions for Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Margin-based loss functions are particularly useful for binary
classification. In contrast to the distance-based losses, these
do not care about the difference between true target and
prediction. Instead they penalize predictions based on how well
they agree with the sign of the target.

+----------------------------------------------+----------------------------------------------------------------------------------------+
| .. toctree::                                 | .. image:: https://rawgithub.com/JuliaML/FileStorage/master/LossFunctions/margin.svg   |
|    :maxdepth: 2                              |                                                                                        |
|                                              |                                                                                        |
|    losses/margin                             |                                                                                        |
+----------------------------------------------+----------------------------------------------------------------------------------------+


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

