Background and Motivation
===========================

In this section we will discuss the concept "loss function" in
more detail. We will start by introducing some terminology and
definitions. However, please note that we won't attempt to give a
complete treatment of loss functions and the math involved
(unlike a book or a lecture could do). So this section won't be a
substitution for proper literature on the topic. While we will
try to cover all the basics necessary to get a decent intuition
of the ideas involved, we do assume basic knowledge about Machine
Learning.

.. warning::

   This section and its sub-sections serve soley as to explain
   the underyling theory and concepts and further to motivate the
   solution provided by this package. As such, this section is
   **not** intended as a guide on how to apply this package.


Terminology
----------------------

To start off, let us go over some basic terminology. In **Machine
Learning** (ML) we are primarily interested in automatically
learning meaningful patterns from data. For our purposes it
suffices to say that in ML we try to teach the computer to solve
a task by induction rather than by definition. This package is
primarily concerned with the subset of Machine Learning that
falls under the umbrella of **Supervised Learning**. There we are
interested in teaching the computer to predict a specific output
for some given input. In contrast to unsupervised learning the
teaching process here involves showing the computer what the
predicted output is supposed to be; i.e. the "true answer" if you
will.

How is this relevant for this package? Well, it implies that we
require some meaningful way to show the true answers to the
computer so that it can learn from "seeing" them. More
importantly, we have to somehow put the true answer into relation
to what the computer currently predicts the answer should be.
This would provide the basic information needed for the computer
to be able to improve; that is what loss functions are for.

When we say we want our computer to learn something that is able
to make predictions, we are talking about a **prediction
function**, denoted as :math:`h` and sometimes called "fitted
hypothesis", or "fitted model". Note that we will avoid the term
hypothesis for the simple reason that it is widely used in
statistics for something completely different. We don't consider
a prediction *function* as the same thing as a prediction
*model*, because we think of a **prediction model** as a family
of prediction functions. What that boils down to is that the
prediction model represents the set of possible prediction
functions, while the final prediction function is the chosen
function that best solves the problem. So in a way a prediction
model can be thought of as the manifestation of our assumptions
about the problem, because it restricts the solution to a
specific family of functions.  For example a linear prediction
model for two features represents all possible linear functions
that have two coefficients. A prediction function would in that
scenario be a concrete linear function with a particular fixed
set of coefficients.

The purpose of a prediction function is to take some input and
produce a corresponding output. That output should be as faithful
as possible to the true answer. In the context of this package we
will refer to the "true answer" as the **true target**, or short
"target". During training, and only during training, inputs and
targets can both be considered as part of our data set. We say
"only during training" because in a production setting we don't
actually have the targets available to us (otherwise there would
be no prediction problem to solve in the first place). In essence
we can think of our data as two entities with a 1-to-1 connection
in each observation, the inputs, which we call **features**, and
the corresponding desired outputs, which we call **true targets**.

Let us be a little more concrete with the two terms we really
care about in this package.

True Targets
    A true target (singular) represents the "desired" output for
    the input features of the observation. The targets are often
    referred to as "ground truth" and we will denote a single
    target as :math:`y \in Y`. When we talk about an array (e.g.
    a vector) of targets, we will print it in bold as
    :math:`\mathbf{y}`. What the set :math:`Y` is will depend on
    the subdomain of supervised learning that you are working in.

    - Real-valued Regression: :math:`Y \subseteq \mathbb{R}`.

    - Multioutput Regression: :math:`Y \subseteq \mathbb{R}^k`.

    - Margin-based Classification: :math:`Y = \{1,-1\}`.

    - Probabilistic Classification: :math:`Y = \{1,0\}`.

    - Multiclass Classification: :math:`Y = \{1,2,\dots,k\}`

    See `MLLabelUtils
    <http://mllabelutilsjl.readthedocs.io/en/latest/api/targets.html>`_
    for more information on classification targets.

Predicted Outputs
    A predicted output (singular) is the result of our prediction
    function given the features of some observation. We will
    denote a single output as :math:`\hat{y} \in \mathbb{R}`
    (pronounced as "why hat"). When we talk about an array of
    outputs, we will print it in bold as
    :math:`\mathbf{\hat{y}}`. Note something unintuitive but
    important: The variables :math:`y` and :math:`\hat{y}` don't
    have to be of the same set. Even in a classification setting
    where :math:`y \in \{1,-1\}`, it is typical that
    :math:`\hat{y} \in \mathbb{R}`.

    The fact that in classification the predictions can be
    fundamentally different than the targets is important to
    know. The reason for restricting the targets to specific
    numbers when doing classification is mathematical convenience
    for loss functions. So loss functions have this knowledge
    build in.

In a classification setting, the predicted outputs and the true
targets are usually of different form and type. For example, in
margin-based classification it could be the case that the target
:math:`y=-1` and the predicted output :math:`\hat{y} = -1000`. It
would seem that the prediction is not really reflecting the
target properly, but in this case we would actually have a
perfectly correct prediction. This is because in margin-based
classification the main thing that matters about the predicted
output is that the sign agrees with the true target.

..  More generally speaking, to be able to directly compare the
    predicted outputs to the targets in a classification setting, one
    first has to convert the predictions into the same form as the
    targets. When doing this, we say that we **classify** the
    prediction. We often refer to the initial predictions that are
    not yet classified as **raw predictions**.

Definitions
----------------------

We base most of our definitions on the work presented in
[STEINWART2008]_. Note, however, that we will adapt or simplify
in places at our discretion. We do this in situations where it
makes sense to us considering the scope of this package or
because of implementation details.

Let us again consider the term **prediction function**. More
formally, a prediction function :math:`h` is a function that maps
an input from the feature space :math:`X` to the real numbers
:math:`\mathbb{R}`. So invoking :math:`h` with some features
:math:`x \in X` will produce the prediction :math:`\hat{y} \in
\mathbb{R}`.

.. math::

   h : X \rightarrow \mathbb{R}

This resulting prediction :math:`\hat{y}` is what we want to
compare to the target :math:`y` in order to asses how bad the
prediction is. The function we use for such an assessment will be
of a family of functions we refer to as supervised losses. We
think of a **supervised loss** as a function of two parameters,
the true target :math:`y \in Y` and the predicted output
:math:`\hat{y} \in \mathbb{R}`. The result of computing such a
loss will be a non-negative real number. The larger the value of
the loss, the worse the prediction.

.. math::

   L : Y \times \mathbb{R} \rightarrow [0,\infty)

Note a few interesting things about supervised loss functions.

- The absolute value of a loss is often (but not always)
  meaningless and doesn't offer itself to a useful
  interpretation. What we usually care about is that the loss is
  as small as it can be.

- In general the loss function we use is not the function we are
  actually interested in minimizing. Instead we are minimizing
  what is referred to as a "surrogate". For binary classification
  for example we are really interested in minimizing the ZeroOne
  loss (which simply counts the number of misclassified
  predictions). However, that loss is difficult to minimize given
  that it is not convex nor continuous. That is why we use other
  loss functions, such as the hinge loss or logistic loss. Those
  losses are "classification calibrated", which basically means
  they are good enough surrogates to solve the same problem.
  Additionally, surrogate losses tend to have other nice
  properties.

- For classification it does not need to be the case that a
  "correct" prediction has a loss of zero. In fact some
  classification calibrated losses are never truly zero.


Alternative Viewpoints
------------------------

While the term "loss function" is usually used in the same
context throughout the literature, the specifics differ from one
textbook to another. For that reason we would like to mention
alternative definitions of what a "loss function" is. Note that
we will only give a partial and thus very simplified description
of these. Please refer to the listed sources for more specifics.

In [SHALEV2014]_ the authors consider a loss function as a
higher-order function of two parameters, a prediction model and
an observation tuple. So in that definition a loss function and
the prediction function are tightly coupled. This way of thinking
about it makes a lot of sense, considering the process of how a
prediction model is usually fit to the data. For gradient descent
to do its job it needs the, well, gradient of the empirical risk.
This gradient is computed using the chain rule for the inner loss
and the prediction model. If one views the loss and the
prediction model as one entity, then the gradient can sometimes
be simplified immensely. That said, we chose to not follow this
school of thought, because from a software-engineering standpoint
it made more sense to us to have small modular pieces. So in our
implementation the loss functions don't need to know that
prediction functions even exist. This makes the package easier to
maintain, test, and reason with. Given Julia's ability for
multiple dispatch we don't even lose the ability to simplify the
gradient if need be.

.. [SHALEV2014] Shalev-Shwartz, Shai, and Shai Ben-David. `"Understanding machine learning: From theory to algorithms" <http://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning>`_. Cambridge University Press, 2014.

