Working with Losses
=====================

Even though they are called loss "functions", this package
implements them as immutable types instead of true Julia
functions. There are good reasons for that. For example it allows
us to specify the properties of losse functions explicitly (e.g.
``isconvex(myloss)``). It also makes for a more consistent API
when it comes to computing the value or the derivative. Some loss
functions even have additional parameters that need to be
specified, such as the :math:`\epsilon` in the case of the
:math:`\epsilon`-insensitive loss. Here, types allow for member
variables to hide that information away from the method
signatures.

In order to avoid potential confusions with true Julia functions,
we will refer to "loss functions" as "losses" instead. The
available losses share a common interface for the most part. This
section will provide an overview of the basic functionality that
is available for all the different types of losses. We will
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

Even more interesting in the example above, is that for such
losses as :class:`L2DistLoss`, which do not have any constructor
parameters or member variables, there is no additional code
executed at all. Such singletons are only used for dispatch and
don't even produce any additional code, which you can observe for
yourself in the code below. As such they are zero-cost
abstractions.

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
comparable to whole families of losses instead of just a single
one. For example, the immutable type :class:`L1EpsilonInsLoss`
has a free parameter :math:`\epsilon`. Each concrete
:math:`\epsilon` results in a different concrete loss of the same
family of epsilon-insensitive losses.

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
single observations under the hood. The core function to compute
the value of a loss is :func:`value`. We will see throughout the
documentation that it allows for a lot of different method
signatures to accomplish a variety of tasks.

.. function:: value(loss, target, output) -> Number

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

.. function:: value(loss, targets, outputs) -> Array

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

   In the case that the two parameters, `targets` and `outputs`,
   are arrays with a different number of dimensions, broadcast
   will be performed. Note that the given parameters are expected
   to have the same size in the dimensions they share.

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



Computing the 1st Derivatives
-------------------------------

Maybe the more interesting aspect of loss functions are their
derivatives. In fact, most of the popular learning algorithm in
Supervised Learning, such as gradient descent, utilize the
derivatives of the loss in one way or the other during the
training process.

To compute the derivative of some loss we expose the function
:func:`deriv`. It supports the same exact method signatures as
:func:`value`. It may be interesting to note explicitly, that we
always compute the derivative in respect to the predicted
``output``, since we are interested in deducing in which
direction the output should change.

.. function:: deriv(loss, target, output) -> Number

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

While broadcast is supported, we do expose a vectorized method
natively. This is done mainly for API consistency reasons.
Internally it even uses broadcast itself, but it does provide the
additional benefit of a more reliable type-inference.

.. function:: deriv(loss, targets, outputs) -> Array

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
            all elements in `targets` and `outputs`.

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

.. function:: value_deriv(loss, target, output) -> Tuple

   Returns the results of :func:`value` and :func:`deriv` as a
   tuple. In some cases this function can yield better
   performance, because the losses can make use of shared
   variables when computing the results. Note that `target` and
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
            the given parameters. They are returned as a Tuple in
            which the first element is the value and the second
            element the derivative.

.. code-block:: jlcon

   #                     loss         y    ŷ
   julia> value_deriv(L2DistLoss(), -1.0, 3.0)
   (16.0,8.0)


Computing the 2nd Derivatives
-------------------------------

Additionally to the first derivative, we also provide the
corresponding methods for the second derivative through the
function :func:`deriv2`. Note again, that we always compute the
derivative in respect to the predicted ``output``.

.. function:: deriv2(loss, target, output) -> Number

   Computes the second derivative for the loss-function denoted
   by the parameter `loss` in respect to the `output`. Note that
   `target` and `output` can be of different numeric type, in
   which case promotion is performed in the manner appropriate
   for the given loss.

   Note: This function should always be type-stable. If it isn't,
   you likely found a bug.

   :param loss: The loss-function :math:`L` we want to compute the
                second derivative with.
   :type loss: :class:`SupervisedLoss`
   :param Number target: The ground truth :math:`y \in Y` of the
                         observation.
   :param Number output: The predicted output :math:`\hat{y} \in
                         \mathbb{R}` for the observation.
   :return: The second derivative of the loss-function for the given
            parameters.

.. code-block:: jlcon

   #               loss             y    ŷ
   julia> deriv2(LogitDistLoss(), -0.5, 0.3)
   0.42781939304058886

   julia> deriv2(LogitMarginLoss(), -1f0, 2f0)
   0.104993574f0

Just like :func:`deriv` and :func:`value`, this function also
supports broadcasting and all the syntax benefits that come with
it. Thus, one can make use of preallocated memory for storing the
element-wise derivatives.

.. code-block:: jlcon

   julia> deriv2.(LogitDistLoss(), [-0.5, 1.2, 3], [0.3, 2.3, -2])
   3-element Array{Float64,1}:
    0.427819
    0.37474
    0.0132961

   julia> buffer = zeros(3); # preallocate a buffer

   julia> buffer .= deriv2.(LogitDistLoss(), [-0.5, 1.2, 3], [0.3, 2.3, -2])
   3-element Array{Float64,1}:
    0.427819
    0.37474
    0.0132961

Furthermore :func:`deriv2` supports all the same method
signatures as :func:`deriv` does. So to avoid repeating the same
text over and over again, please look at the documentation of
:func:`deriv` for more information.


Function Closures
---------------------

In some circumstances it may be convenient to have the loss function
or its derivative as a proper Julia function. Instead of
exporting special function names for every implemented loss (like
``l2distloss(...)``), we provide the ability to generate a true
function on the fly for any given loss.

.. function:: value_fun(loss) -> Function

   Returns a new function that computes the :func:`value` for the
   given `loss`. This new function will support all the signatures
   that :func:`value` does.

   :param Loss loss: The loss we want the function for.

.. code-block:: jlcon

   julia> f = value_fun(L2DistLoss())
   (::_value) (generic function with 1 method)

   julia> f(-1.0, 3.0) # computes the value of L2DistLoss
   16.0

   julia> f.([1.,2], [4,7])
   2-element Array{Float64,1}:
     9.0
    25.0


.. function:: deriv_fun(loss) -> Function

   Returns a new function that computes the :func:`deriv` for the
   given `loss`. This new function will support all the signatures
   that :func:`deriv` does.

   :param Loss loss: The loss we want the derivative-function for.

.. code-block:: julia

   julia> g = deriv_fun(L2DistLoss())
   (::_deriv) (generic function with 1 method)

   julia> g(-1.0, 3.0) # computes the deriv of L2DistLoss
   8.0

   julia> g.([1.,2], [4,7])
   2-element Array{Float64,1}:
     6.0
    10.0


.. function:: deriv2_fun(loss) -> Function

   Returns a new function that computes the :func:`deriv2` (i.e.
   second derivative) for the given `loss`. This new function
   will support all the signatures that :func:`deriv2` does.

   :param Loss loss: The loss we want the second-derivative
                     function for.

.. code-block:: julia

   julia> g2 = deriv2_fun(L2DistLoss())
   (::_deriv2) (generic function with 1 method)

   julia> g2(-1.0, 3.0) # computes the second derivative of L2DistLoss
   2.0

   julia> g2.([1.,2], [4,7])
   2-element Array{Float64,1}:
    2.0
    2.0


.. function:: value_deriv_fun(loss) -> Function

   Returns a new function that computes the :func:`value_deriv`
   for the given `loss`. This new function will support all the
   signatures that :func:`value_deriv` does.

   :param Loss loss: The loss we want the function for.

.. code-block:: julia

   julia> fg = value_deriv_fun(L2DistLoss())
   (::_value_deriv) (generic function with 1 method)

   julia> fg(-1.0, 3.0) # computes the second derivative of L2DistLoss
   (16.0,8.0)


Note, however, that these closures cause quite an overhead when
executed in the global scope. If you want to use them
efficiently, either don't create them in global scope, or make
sure that you pass the closure to some other function before it
is used. This way the compiler will most likely inline it and it
will be a zero cost abstraction.

.. code-block:: jlcon

   julia> f = value_fun(L2DistLoss())
   (::_value) (generic function with 1 method)

   julia> @code_llvm f(-1.0, 3.0)
   define %jl_value_t* @julia__value_70960(%jl_value_t*, %jl_value_t**, i32) #0 {
   top:
     %3 = alloca %jl_value_t**, align 8
     store volatile %jl_value_t** %1, %jl_value_t*** %3, align 8
     %ptls_i8 = call i8* asm "movq %fs:0, $0;\0Aaddq $$-2672, $0", "=r,~{dirflag},~{fpsr},~{flags}"() #2
       [... many more lines of code ...]
     %15 = call %jl_value_t* @jl_f__apply(%jl_value_t* null, %jl_value_t** %5, i32 3)
     %16 = load i64, i64* %11, align 8
     store i64 %16, i64* %9, align 8
     ret %jl_value_t* %15
   }

   julia> foo(t,y) = (f = value_fun(L2DistLoss()); f(t,y))
   foo (generic function with 1 method)

   julia> @code_llvm foo(-1.0, 3.0)
   define double @julia_foo_71242(double, double) #0 {
   top:
     %2 = fsub double %1, %0
     %3 = fmul double %2, %2
     ret double %3
   }


Properties of a Loss
------------------------

In some situations it can be quite useful to assert certain
properties about a loss-function. One such scenario could be when
implementing an algorithm that requires the loss to be strictly
convex or Lipschitz continuous.

This package uses functions to represent individual properties of
a loss. It follows a list of implemented property functions
defined in `LearnBase.jl
<https://github.com/JuliaML/LearnBase.jl>`_.

.. function:: isconvex(loss) -> Bool

   Returns true if given loss is a convex function.
   A function :math:`f : \mathbb{R}^n \rightarrow \mathbb{R}` is convex
   if **dom f** is a convex set and if :math:`\forall` x, y in the domain,
   and :math:`\theta` such that for :math:`0` :math:`\le` :math:`\theta` :math:`\le` :math:`1` , we have

   .. math:: f(\theta x + (1 - \theta)y) \le \theta f(x) + (1 - \theta) f(y)

   For more about convex functions, check `this<https://en.wikipedia.org/wiki/Convex_function>`_.

   :param Loss loss: The loss we want to check for convexity.

.. code-block:: julia

   julia> isconvex(LPDistLoss(0.1))
   false

   julia> isconvex(LPDistLoss(2))
   true


.. function:: isstrictlyconvex(loss) -> Bool

   Returns true if given loss is a strictly convex function.
   A function :math:`f : \mathbb{R}^n \rightarrow \mathbb{R}` is strictly convex
   if **domain f** is a convex set and if :math:`\forall` x :math:`\neq` y in the domain,
   and :math:`\theta` such that for :math:`0 <` :math:`\theta` :math:`< 1` , we have

   .. math:: f(\theta x + (1 - \theta)y) < \theta f(x) + (1 - \theta) f(y)

   For more about convex functions, check `this<https://en.wikipedia.org/wiki/Convex_function>`_.

   :param Loss loss: The loss we want to check for strict convexity.

.. code-block:: julia

   julia> isstrictlyconvex(LPDistLoss(0.1))
   false

   julia> isstrictlyconvex(LPDistLoss(2))
   true

.. function:: isstronglyconvex(loss) -> Bool

   Returns true if given loss is a strongly convex function.
   A function :math:`f : \mathbb{R}^n \rightarrow \mathbb{R}` is :math:`m-`strongly convex
   if **domain f** is a convex set and if :math:`\forall` x :math:`\neq` y in the domain,
   and :math:`\theta` such that for :math:`0` :math:`\le` :math:`\theta` :math:`\le` :math:`1` , we have

   .. math:: f(\theta x + (1 - \theta)y) < \theta f(x) + (1 - \theta) f(y) - 0.5 m \theta (1 - \theta) {{|| x - y ||}^{2}}_{2}

   In a more familiar setting, if the loss function is differentiable we have

   .. math:: (\grad f(x) - \grad f(y) )^{T} (x - y) \ge m {{|| x - y||}^{2}}_{2}

   For more about convex functions, check `this<https://en.wikipedia.org/wiki/Convex_function>`_.

   :param Loss loss: The loss we want to check for strong convexity.

.. code-block:: julia

   julia> isstronglyconvex(LPDistLoss(0.1))
   false

   julia> isstronglyconvex(LPDistLoss(2))
   true

.. function:: isdifferentiable(loss[, at]) -> Bool

   Returns true if given loss is a differentiable function.
   A function :math:`f : \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}` is differentiable at a
   point x in **int domain f** if there exists a row vector :math:`Df(x)` in :math:`\mathbb{R}^{mxn}`
   such that the following limit exists

   .. math:: \lim_{z \neq x, z \to x} \frac{{||f(z) - f(x) - Df(x)(z-x)||}_2}{{||z - x||}_2}

   For more about differentiable functions, check `this<https://en.wikipedia.org/wiki/Differentiable_function>`_.

   :param Loss loss: The loss we want to check for differentiability.

.. code-block:: julia

   julia> isdifferentiable(LPDistLoss(1))
   false

   julia> isdifferentiable(LPDistLoss(2))
   true

.. function:: istwicedifferentiable(loss[, at]) -> Bool

   Returns true if given loss is a twice differentiable function.
   A function :math:`f : \mathbb{R}^{n} \rightarrow \mathbb{R}` is said to be twice differentiable at a point x in
   **int domain f** if the function derivative for \grad f exists at x.

   .. math:: \grad^2 f(x) = D \grad f(x)

   For more about differentiable functions, check `this<https://en.wikipedia.org/wiki/Differentiable_function>`_.

   :param Loss loss: The loss we want to check for differentiability.

.. code-block:: julia

   julia> istwicedifferentiable(LPDistLoss(1))
   false

   julia> istwicedifferentiable(LPDistLoss(2))
   true

.. function:: isnemitski(loss) -> Bool

   Returns true if given loss is a Nemitski loss function.

   A given loss function :math:`L : X \times Y \times \mathbb{R} \rightarrow [0,\infty)` is called a Nemitski loss
   if there exist a measurable function :math:`b : X \times Y \rightarrow [0, \infty)` and an increasing
   function :math:`h : [0, \infty) \rightarrow [0, \infty) such that

   .. math:: L(x,y,t) \le b(x,y) + h(|t|),  where (x,y,t)  \in X \times Y \times \mathbb{R}.

   Furthermore, we say that L is a Nemitski loss of order :math:`p \in (0, \infty)` if there exists a constant c > 0
   such that

   .. math:: L(x,y,t) \le b(x,y) + c|t|^{p} , (x,y,t) \in X \times Y \times \mathbb{R}.

   :param Loss loss: The loss we want to check for the Nemitski condition.

.. code-block:: julia

    julia> isnemitski(LPDistLoss(0.2))
    false

    julia> isnemitski(LPDistLoss(2))
    true

.. function:: islipschitzcont(loss) -> Bool

    Returns true if given loss function is Lipschitz continuous.

    A loss function :math:`L : X \times Y \times \mathbb{R} \rightarrow [0, \infty)` is
    Lipschitz continous if there exits a finite constant :math:`M < \infty` such that

    .. math:: |L(x, y, t) - L(x, y, t')| \le M |t - t'| \forall (x, y, t) \in X \times Y \times \mathbb{R}

    For more about Lipschitz-continuity check `this<https://en.wikipedia.org/wiki/Lipschitz_continuity>`_.

    :param Loss loss: The loss we want to check for being Lipschitz continuous.

.. code-block:: julia

    julia> islipschitzcont(SigmoidLoss())
    true

    julia> islipschitzcont(ExpLoss())
    false

.. function:: islocallylipschitzcont(loss) -> Bool

    Returns true if given loss function is locally-Lipschitz continous.

    A loss :math:`L : X \times Y \times \mathbb{R} \rightarrow [0, \infty)` is called locally Lipschitz
    continuous if :math:`\forall a \ge 0` there exists a constant :math:`c_a \ge 0` such that

    .. math:: sup_{x \in X, y \in Y} |L(x,y,t) − L(x,y,t′)| \le c_a |t − t′|, where t,t′ \in [−a,a]

    For more about locally Lipschitz-continuity check `this<https://en.wikipedia.org/wiki/Lipschitz_continuity>`_.

    :param Loss loss: The loss we want to check for being locally Lipschitz-continous.

.. code-block:: julia

    julia> islocallylipschitzcont(ExpLoss())
    true

    julia> islocallylipschitzcont(SigmoidLoss())
    true

.. function:: isclipable(loss) -> Bool

    Returns true if given loss function is clipable.

    A loss :math:`L : X \times Y \times \mathbb{R} \rightarrow [0, \infty)` can be clipped at M > 0 if,
    for all (x,y,t) :math:`\in X \times Y \times \mathbb{R},

    .. math:: L(x, y, \hat{t}) \le L(x, y, t)

    where \hat{t} denotes the clipped value of t at ±M. That is

    .. math:: \hat{t} = \begin{cases} -M & \quad \text{if } t < -M \\ t & \quad \text{if } t \in [-M, M] \\ M & \quad \text{if } t > M \end{cases}

    :param Loss loss: The loss we want to check for being clipable.

.. code-block:: julia

    julia> isclipable(ExpLoss())
    false

    julia> isclipable(L2DistLoss())
    true

.. function:: ismarginbased(loss) -> Bool

   Returns true if given loss is a Distance-based Loss.

   A Supervised Loss function :math:`f : \mathbb{R} \times \mathbb{R} \rightarrow [0, \infty)`
   is said to be **margin-based** if there exists a representing function
   :math:`\psi : \mathbb{R} \rightarrow [0, \infty)` satisfying

   ..math:: L(y, t) = \psi (yt), where y \in Y, t \in \mathbb{R}

   :param Loss loss: The loss we want to check for being Margin-based.

.. code-block:: julia

   julia> ismarginbased(HuberLoss(2))
   false

   julia> ismarginbased(L2MarginLoss())
   true

.. function:: isclasscalibrated(loss) -> Bool

.. function:: isdistancebased(loss) -> Bool

   Returns true if given loss is a Distance-based Loss.

   A Supervised Loss function :math:`f : \mathbb{R} \times \mathbb{R} \rightarrow [0, \infty)`
   is said to be **distance-based** if there exists a representing function
   :math:`\psi : \mathbb{R} \rightarrow [0, \infty)` satisfying :math:`\psi (0) = 0` and

   ..math:: L(y, t) = \psi (y - t), where y \in Y, t \in \mathbb{R}

   :param Loss loss: The loss we want to check for being Distance-based.

.. code-block:: julia

   julia> isdistancebased(HuberLoss(2))
   true

   julia> isdistancebased(L2MarginLoss())
   false

.. function:: issymmetric(loss) -> Bool

   Returns true if given loss is a Symmetric Loss.

   A function :math:`f : \mathbb{R} \rightarrow [0,\infty)` is said to be
   symmetric about origin if we have

   ..math:: f(x) = f(-x) \forall x \in \mathbb{R}

   :param Loss loss: The loss we want to check for being symmetric.

.. code-block:: julia

   julia> issymmetric(QuantileLoss(0.2))
   false

   julia> issymetric(LPDistLoss(2))
   true
