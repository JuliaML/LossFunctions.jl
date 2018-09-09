var documenterSearchIndex = {"docs": [

{
    "location": "#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "#LossFunctions.jl\'s-documentation-1",
    "page": "Home",
    "title": "LossFunctions.jl\'s documentation",
    "category": "section",
    "text": "This package represents a community effort to centralize the definition and implementation of loss functions in Julia. As such, it is a part of the JuliaML ecosystem.The sole purpose of this package is to provide an efficient and extensible implementation of various loss functions used throughout Machine Learning (ML). It is thus intended to serve as a special purpose back-end for other ML libraries that require losses to accomplish their tasks. To that end we provide a considerable amount of carefully implemented loss functions, as well as an API to query their properties (e.g. convexity). Furthermore, we expose methods to compute their values, derivatives, and second derivatives for single observations as well as arbitrarily sized arrays of observations. In the case of arrays a user additionally has the ability to define if and how element-wise results are averaged or summed over.From an end-user\'s perspective one normally does not need to import this package directly. That said, it should provide a decent starting point for any student that is interested in investigating the properties or behaviour of loss functions."
},

{
    "location": "#Introduction-and-Motivation-1",
    "page": "Home",
    "title": "Introduction and Motivation",
    "category": "section",
    "text": "If this is the first time you consider using LossFunctions for your machine learning related experiments or packages, make sure to check out the \"Getting Started\" section.Pages = [\"introduction/gettingstarted.md\"]\nDepth = 2If you are new to Machine Learning in Julia, or are simply interested in how and why this package works the way it works, feel free to take a look at the following sections. There we discuss the concepts involved and outline the most important terms and definitions.Pages = [\"introduction/motivation.md\"]\nDepth = 2"
},

{
    "location": "#User\'s-Guide-1",
    "page": "Home",
    "title": "User\'s Guide",
    "category": "section",
    "text": "This section gives a more detailed treatment of the exposed functions and their available methods. We will start by describing how to instantiate a loss, as well as the basic interface that all loss functions share.Pages = [\"user/interface.md\"]\nDepth = 2Next we will consider how to average or sum the results of the loss functions more efficiently. The methods described here are implemented in such a way as to avoid allocating a temporary array.Pages = [\"user/aggregate.md\"]\nDepth = 2"
},

{
    "location": "#Available-Losses-1",
    "page": "Home",
    "title": "Available Losses",
    "category": "section",
    "text": "Aside from the interface, this package also provides a number of popular (and not so popular) loss functions out-of-the-box. Great effort has been put into ensuring a correct, efficient, and type-stable implementation for those. Most of them either belong to the family of distance-based or margin-based losses. These two categories are also indicative for if a loss is intended for regression or classification problems"
},

{
    "location": "#Loss-Functions-for-Regression-1",
    "page": "Home",
    "title": "Loss Functions for Regression",
    "category": "section",
    "text": "Loss functions that belong to the category \"distance-based\" are primarily used in regression problems. They utilize the numeric difference between the predicted output and the true target as a proxy variable to quantify the quality of individual predictions.<table><tbody><tr><td style=\"text-align: left;\">Pages = [\"losses/distance.md\"]\nDepth = 2</td><td>(Image: distance-based losses)</td></tr></tbody></table>"
},

{
    "location": "#Loss-Functions-for-Classification-1",
    "page": "Home",
    "title": "Loss Functions for Classification",
    "category": "section",
    "text": "Margin-based loss functions are particularly useful for binary classification. In contrast to the distance-based losses, these do not care about the difference between true target and prediction. Instead they penalize predictions based on how well they agree with the sign of the target.<table><tbody><tr><td style=\"text-align: left;\">Pages = [\"losses/margin.md\"]\nDepth = 2</td><td>(Image: margin-based losses)</td></tr></tbody></table>"
},

{
    "location": "#Advanced-Topics-1",
    "page": "Home",
    "title": "Advanced Topics",
    "category": "section",
    "text": "In some situations it can be useful to slightly alter an existing loss function. We provide two general ways to accomplish that. The first way is to scale a loss by a constant factor. This can for example be useful to transform the L2DistLoss into the least squares loss one knows from statistics. The second way is to reweight the two classes of a binary classification loss. This is useful for handling inbalanced class distributions.Pages = [\"advanced/extend.md\"]\nDepth = 2If you are interested in contributing to LossFunctions.jl, or simply want to understand how and why the package does then take a look at our developer documentation (although it is a bit sparse at the moment).Pages = [\"advanced/developer.md\"]\nDepth = 2"
},

{
    "location": "#Index-1",
    "page": "Home",
    "title": "Index",
    "category": "section",
    "text": "Pages = [\"indices.md\"]"
},

{
    "location": "introduction/gettingstarted/#",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "page",
    "text": ""
},

{
    "location": "introduction/gettingstarted/#Getting-Started-1",
    "page": "Getting Started",
    "title": "Getting Started",
    "category": "section",
    "text": "LossFunctions.jl is the result of a collaborative effort to design and implement an efficient but also convenient-to-use Julia library for, well, loss functions. As such, this package implements the functionality needed to query various properties about a loss function (such as convexity), as well as a number of methods to compute its value, derivative, and second derivative for single observations or arrays of observations.In this section we will provide a condensed overview of the package. In order to keep this overview concise, we will not discuss any background information or theory on the losses here in detail."
},

{
    "location": "introduction/gettingstarted/#Installation-1",
    "page": "Getting Started",
    "title": "Installation",
    "category": "section",
    "text": "To install LossFunctions.jl, start up Julia and type the following code-snipped into the REPL. It makes use of the native Julia package manger.using Pkg\nPkg.add(\"LossFunctions\")"
},

{
    "location": "introduction/gettingstarted/#Overview-1",
    "page": "Getting Started",
    "title": "Overview",
    "category": "section",
    "text": "Let us take a look at a few examples to get a feeling of how one can use this library. This package is registered in the Julia package ecosystem. Once installed the package can be imported as usual.using LossFunctionsTypically, the losses we work with in Machine Learning are multivariate functions of two variables, the true target y, which represents the \"ground truth\" (i.e. correct answer), and the predicted output haty, which is what our model thinks the truth is. All losses that can be expressed in this way will be referred to as supervised losses. The true targets are often expected to be of a specific set (e.g. 1-1 in classification), which we will refer to as Y, while the predicted outputs may be any real number. So for our purposes we can define a supervised loss as followsL  Y times mathbbR rightarrow 0infty)Such a loss function takes these two variables as input and returns a value that quantifies how \"bad\" our prediction is in comparison to the truth. In other words: the lower the loss, the better the prediction.From an implementation perspective, we should point out that all the concrete loss \"functions\" that this package provides are actually defined as immutable types, instead of native Julia functions. We can compute the value of some type of loss using the function value. Let us start with an example of how to compute the loss of a single observation (i.e. two numbers).#                loss       y    ŷ\njulia> value(L2DistLoss(), 1.0, 0.5)\n0.25Calling the same function using arrays instead of numbers will return the element-wise results, and thus basically just serve as a wrapper for broadcast (which by the way is also supported).julia> true_targets = [  1,  0, -2];\n\njulia> pred_outputs = [0.5,  2, -1];\n\njulia> value(L2DistLoss(), true_targets, pred_outputs)\n3-element Array{Float64,1}:\n 0.25\n 4.0\n 1.0Alternatively, one can also use an instance of a loss just like one would use any other Julia function. This can make the code significantly more readable while not impacting performance, as it is a zero-cost abstraction (i.e. it compiles down to the same code).julia> loss = L2DistLoss()\nLossFunctions.LPDistLoss{2}()\n\njulia> loss(true_targets, pred_outputs) # same result as above\n3-element Array{Float64,1}:\n 0.25\n 4.0\n 1.0\n\njulia> loss(1, 0.5f0) # single observation\n0.25f0If you are not actually interested in the element-wise results individually, but some accumulation of those (such as mean or sum), you can additionally specify an average mode. This will avoid allocating a temporary array and directly compute the result.julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.Sum())\n5.25\n\njulia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.Mean())\n1.75Aside from these standard unweighted average modes, we also provide weighted alternatives. These expect a weight-factor for each observation in the predicted outputs and so allow to give certain observations a stronger influence over the result.julia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.WeightedSum([2,1,1]))\n5.5\n\njulia> value(L2DistLoss(), true_targets, pred_outputs, AvgMode.WeightedMean([2,1,1]))\n1.375We do not restrict the targets and outputs to be vectors, but instead allow them to be arrays of any arbitrary shape. The shape of an array may or may not have an interpretation that is relevant for computing the loss. Consequently, those methods that don\'t require this information can be invoked using the same method signature as before, because the results are simply computed element-wise or accumulated.julia> A = rand(2,3)\n2×3 Array{Float64,2}:\n 0.0939946  0.97639   0.568107\n 0.183244   0.854832  0.962534\n\njulia> B = rand(2,3)\n2×3 Array{Float64,2}:\n 0.0538206  0.77055  0.996922\n 0.598317   0.72043  0.912274\n\njulia> value(L2DistLoss(), A, B)\n2×3 Array{Float64,2}:\n 0.00161395  0.0423701  0.183882\n 0.172286    0.0180639  0.00252607\n\njulia> value(L2DistLoss(), A, B, AvgMode.Sum())\n0.420741920634These methods even allow arrays of different dimensionality, in which case broadcast is performed. This also applies to computing the sum and mean, in which case we use custom broadcast implementations that avoid allocating a temporary array.julia> value(L2DistLoss(), rand(2), rand(2,2))\n2×2 Array{Float64,2}:\n 0.228077  0.597212\n 0.789808  0.311914\n\njulia> value(L2DistLoss(), rand(2), rand(2,2), AvgMode.Sum())\n0.0860658081865589That said, it is possible to explicitly specify which dimension denotes the observations. This is particularly useful for multivariate regression where one could want to accumulate the loss per individual observation.julia> value(L2DistLoss(), A, B, AvgMode.Sum(), ObsDim.First())\n2-element Array{Float64,1}:\n 0.227866\n 0.192876\n\njulia> value(L2DistLoss(), A, B, AvgMode.Sum(), ObsDim.Last())\n3-element Array{Float64,1}:\n 0.1739\n 0.060434\n 0.186408\n\njulia> value(L2DistLoss(), A, B, AvgMode.WeightedSum([2,1]), ObsDim.First())\n0.648608280735All these function signatures of value also apply for computing the derivatives using deriv and the second derivatives using deriv2.julia> true_targets = [  1,  0, -2];\n\njulia> pred_outputs = [0.5,  2, -1];\n\njulia> deriv(L2DistLoss(), true_targets, pred_outputs)\n3-element Array{Float64,1}:\n -1.0\n  4.0\n  2.0\n\njulia> deriv2(L2DistLoss(), true_targets, pred_outputs)\n3-element Array{Float64,1}:\n 2.0\n 2.0\n 2.0Additionally, we provide mutating versions for the subset of methods that return an array. These have the same function signatures with the only difference of requiring an additional parameter as the first argument. This variable should always be the preallocated array that is to be used as storage.julia> buffer = zeros(3)\n3-element Array{Float64,1}:\n 0.0\n 0.0\n 0.0\n\njulia> deriv!(buffer, L2DistLoss(), true_targets, pred_outputs)\n3-element Array{Float64,1}:\n -1.0\n  4.0\n  2.0"
},

{
    "location": "introduction/gettingstarted/#Getting-Help-1",
    "page": "Getting Started",
    "title": "Getting Help",
    "category": "section",
    "text": "To get help on specific functionality you can either look up the information here, or if you prefer you can make use of Julia\'s native doc-system. The following example shows how to get additional information on L1HingeLoss within Julia\'s REPL:?L1HingeLosssearch: L1HingeLoss SmoothedL1HingeLoss\n\n  L1HingeLoss <: MarginLoss\n\n  The hinge loss linearly penalizes every predicition where the resulting\n  agreement < 1 . It is Lipschitz continuous and convex, but not strictly\n  convex.\n\n  L(y, ŷ) = max(0, 1 - y⋅ŷ)\n\n             Lossfunction                     Derivative\n     ┌────────────┬────────────┐      ┌────────────┬────────────┐\n   3 │\'\\.                      │    0 │                  ┌------│\n     │  \'\'_                    │      │                  |      │\n     │     \\.                  │      │                  |      │\n     │       \'.                │      │                  |      │\n   L │         \'\'_             │   L\' │                  |      │\n     │            \\.           │      │                  |      │\n     │              \'.         │      │                  |      │\n   0 │                \'\'_______│   -1 │------------------┘      │\n     └────────────┴────────────┘      └────────────┴────────────┘\n     -2                        2      -2                        2\n                y ⋅ ŷ                            y ⋅ ŷIf you find yourself stuck or have other questions concerning the package you can find us on the julialang slack or the Machine Learning domain on discourse.julialang.orgMachine Learning on JulialangIf you encounter a bug or would like to participate in the further development of this package come find us on Github.JuliaML/LossFunctions.jl"
},

{
    "location": "introduction/motivation/#",
    "page": "Background and Motivation",
    "title": "Background and Motivation",
    "category": "page",
    "text": ""
},

{
    "location": "introduction/motivation/#Background-and-Motivation-1",
    "page": "Background and Motivation",
    "title": "Background and Motivation",
    "category": "section",
    "text": "In this section we will discuss the concept \"loss function\" in more detail. We will start by introducing some terminology and definitions. However, please note that we won\'t attempt to give a complete treatment of loss functions and the math involved (unlike a book or a lecture could do). So this section won\'t be a substitution for proper literature on the topic. While we will try to cover all the basics necessary to get a decent intuition of the ideas involved, we do assume basic knowledge about Machine Learning.warning: Warning\nThis section and its sub-sections serve soley as to explain the underyling theory and concepts and further to motivate the solution provided by this package. As such, this section is not intended as a guide on how to apply this package."
},

{
    "location": "introduction/motivation/#Terminology-1",
    "page": "Background and Motivation",
    "title": "Terminology",
    "category": "section",
    "text": "To start off, let us go over some basic terminology. In Machine Learning (ML) we are primarily interested in automatically learning meaningful patterns from data. For our purposes it suffices to say that in ML we try to teach the computer to solve a task by induction rather than by definition. This package is primarily concerned with the subset of Machine Learning that falls under the umbrella of Supervised Learning. There we are interested in teaching the computer to predict a specific output for some given input. In contrast to unsupervised learning the teaching process here involves showing the computer what the predicted output is supposed to be; i.e. the \"true answer\" if you will.How is this relevant for this package? Well, it implies that we require some meaningful way to show the true answers to the computer so that it can learn from \"seeing\" them. More importantly, we have to somehow put the true answer into relation to what the computer currently predicts the answer should be. This would provide the basic information needed for the computer to be able to improve; that is what loss functions are for.When we say we want our computer to learn something that is able to make predictions, we are talking about a prediction function, denoted as h and sometimes called \"fitted hypothesis\", or \"fitted model\". Note that we will avoid the term hypothesis for the simple reason that it is widely used in statistics for something completely different. We don\'t consider a prediction function as the same thing as a prediction model, because we think of a prediction model as a family of prediction functions. What that boils down to is that the prediction model represents the set of possible prediction functions, while the final prediction function is the chosen function that best solves the prediction problem. So in a way a prediction model can be thought of as the manifestation of our assumptions about the problem, because it restricts the solution to a specific family of functions. For example a linear prediction model for two features represents all possible linear functions that have two coefficients. A prediction function would in that scenario be a concrete linear function with a particular fixed set of coefficients.The purpose of a prediction function is to take some input and produce a corresponding output. That output should be as faithful as possible to the true answer. In the context of this package we will refer to the \"true answer\" as the true target, or short \"target\". During training, and only during training, inputs and targets can both be considered as part of our data set. We say \"only during training\" because in a production setting we don\'t actually have the targets available to us (otherwise there would be no prediction problem to solve in the first place). In essence we can think of our data as two entities with a 1-to-1 connection in each observation, the inputs, which we call features, and the corresponding desired outputs, which we call true targets.Let us be a little more concrete with the two terms we really care about in this package.True Targets:\nA true target (singular) represents the \"desired\" output for the input features of a single observation. The targets are often referred to as \"ground truth\" and we will denote a single target as y in Y. While y can be a scalar or some array, the key is that it represents the target of a single observation. When we talk about an array (e.g. a vector) of multiple targets, we will print it in bold as mathbfy. What the set Y is will depend on the subdomain of supervised learning that you are working in.\nReal-valued Regression: Y subseteq mathbbR.\nMultioutput Regression: Y subseteq mathbbR^k.\nMargin-based Classification: Y = 1-1.\nProbabilistic Classification: Y = 10.\nMulticlass Classification: Y = 12dotsk.\nSee MLLabelUtils for more information on classification targets.\nPredicted Outputs:\nA predicted output (singular) is the result of our prediction function given the features of some observation. We will denote a single output as haty in mathbbR (pronounced as \"why hat\"). When we talk about an array of outputs for multiple observations, we will print it in bold as mathbfhaty. Note something unintuitive but important: The variables y and haty don\'t have to be of the same set. Even in a classification setting where y in 1-1, it is typical that haty in mathbbR.\nThe fact that in classification the predictions can be fundamentally different than the targets is important to know. The reason for restricting the targets to specific numbers when doing classification is mathematical convenience for loss functions. So loss functions have this knowledge build in.In a classification setting, the predicted outputs and the true targets are usually of different form and type. For example, in margin-based classification it could be the case that the target y=-1 and the predicted output haty = -1000. It would seem that the prediction is not really reflecting the target properly, but in this case we would actually have a perfectly correct prediction. This is because in margin-based classification the main thing that matters about the predicted output is that the sign agrees with the true target.Even though we talked about prediction functions and features, we will see that for computing loss functions all we really care about are the true targets and the predicted outputs, regardless of how the outputs were produced."
},

{
    "location": "introduction/motivation/#Definitions-1",
    "page": "Background and Motivation",
    "title": "Definitions",
    "category": "section",
    "text": "We base most of our definitions on the work presented in [STEINWART2008]. Note, however, that we will adapt or simplify in places at our discretion. We do this in situations where it makes sense to us considering the scope of this package or because of implementation details.Let us again consider the term prediction function. More formally, a prediction function h is a function that maps an input from the feature space X to the real numbers mathbbR. So invoking h with some features x in X will produce the prediction haty in mathbbR.h  X rightarrow mathbbRThis resulting prediction haty is what we want to compare to the target y in order to asses how bad the prediction is. The function we use for such an assessment will be of a family of functions we refer to as supervised losses. We think of a supervised loss as a function of two parameters, the true target y in Y and the predicted output haty in mathbbR. The result of computing such a loss will be a non-negative real number. The larger the value of the loss, the worse the prediction.L  Y times mathbbR rightarrow 0infty)Note a few interesting things about supervised loss functions.The absolute value of a loss is often (but not always) meaningless and doesn\'t offer itself to a useful interpretation. What we usually care about is that the loss is as small as it can be.\nIn general the loss function we use is not the function we are actually interested in minimizing. Instead we are minimizing what is referred to as a \"surrogate\". For binary classification for example we are really interested in minimizing the ZeroOne loss (which simply counts the number of misclassified predictions). However, that loss is difficult to minimize given that it is not convex nor continuous. That is why we use other loss functions, such as the hinge loss or logistic loss. Those losses are \"classification calibrated\", which basically means they are good enough surrogates to solve the same problem. Additionally, surrogate losses tend to have other nice properties.\nFor classification it does not need to be the case that a \"correct\" prediction has a loss of zero. In fact some classification calibrated losses are never truly zero.There are two sub-families of supervised loss-functions that are of particular interest, namely margin-based losses and distance-based losses. These two categories of loss functions are especially useful for the two basic sub-domains of supervised learning: Classification and Regression."
},

{
    "location": "introduction/motivation/#Margin-based-Losses-for-Classification-1",
    "page": "Background and Motivation",
    "title": "Margin-based Losses for Classification",
    "category": "section",
    "text": "Margin-based losses are mainly utilized for binary classification problems where the goal is to predict a categorical value. They assume that the set of targets Y is restricted to Y = 1-1. These two possible values for the target denote the positive class in the case of y = 1, and the negative class in the case of y = -1. In contrast to other formalism, they do not natively provide probabilities as output.More formally, we call a supervised loss function L  Y times mathbbR rightarrow 0 infty) margin-based if there exists a representing function psi  mathbbR rightarrow 0 infty) such thatL(y haty) = psi (y cdot haty)  qquad  y in Y haty in mathbbRnote: Note\nThroughout the codebase we refer to the result of y cdot haty as agreement. The discussion that lead to this convention can be found issue #9"
},

{
    "location": "introduction/motivation/#Distance-based-Losses-for-Regression-1",
    "page": "Background and Motivation",
    "title": "Distance-based Losses for Regression",
    "category": "section",
    "text": "Distance-based losses are usually used in regression settings where the goal is to predict some real valued variable. The goal there is that the prediction is as close as possible to the true target. In such a scenario it is quite sensible to penalize the distance between the prediction and the target in some way.More formally, a supervised loss function L  Y times mathbbR rightarrow 0 infty) is said to be distance-based, if there exists a representing function psi  mathbbR rightarrow 0 infty) satisfying psi (0) = 0 andL(y haty) = psi (haty - y)  qquad  y in Y haty in mathbbRnote: Note\nIn the literature that this package is partially based on, the convention for the distance-based losses is that r = y - haty (see [STEINWART2008] p. 38). We chose to diverge from this definition because it would force a difference of the sign between the results for the unary and the binary version of the derivative. That difference would be a introduced by the chain rule, since the inner derivative would result in fracpartialpartial haty (y - haty) = -1."
},

{
    "location": "introduction/motivation/#Alternative-Viewpoints-1",
    "page": "Background and Motivation",
    "title": "Alternative Viewpoints",
    "category": "section",
    "text": "While the term \"loss function\" is usually used in the same context throughout the literature, the specifics differ from one textbook to another. For that reason we would like to mention alternative definitions of what a \"loss function\" is. Note that we will only give a partial and thus very simplified description of these. Please refer to the listed sources for more specifics.In [SHALEV2014] the authors consider a loss function as a higher-order function of two parameters, a prediction model and an observation tuple. So in that definition a loss function and the prediction function are tightly coupled. This way of thinking about it makes a lot of sense, considering the process of how a prediction model is usually fit to the data. For gradient descent to do its job it needs the, well, gradient of the empirical risk. This gradient is computed using the chain rule for the inner loss and the prediction model. If one views the loss and the prediction model as one entity, then the gradient can sometimes be simplified immensely. That said, we chose to not follow this school of thought, because from a software-engineering standpoint it made more sense to us to have small modular pieces. So in our implementation the loss functions don\'t need to know that prediction functions even exist. This makes the package easier to maintain, test, and reason with. Given Julia\'s ability for multiple dispatch we don\'t even lose the ability to simplify the gradient if need be."
},

{
    "location": "introduction/motivation/#References-1",
    "page": "Background and Motivation",
    "title": "References",
    "category": "section",
    "text": "[STEINWART2008]: Steinwart, Ingo, and Andreas Christmann. \"Support vector machines\". Springer Science & Business Media, 2008.[SHALEV2014]: Shalev-Shwartz, Shai, and Shai Ben-David. \"Understanding machine learning: From theory to algorithms\". Cambridge University Press, 2014."
},

{
    "location": "user/interface/#",
    "page": "Working with Losses",
    "title": "Working with Losses",
    "category": "page",
    "text": "DocTestSetup = quote\n    using LossFunctions\nend"
},

{
    "location": "user/interface/#Working-with-Losses-1",
    "page": "Working with Losses",
    "title": "Working with Losses",
    "category": "section",
    "text": "Even though they are called loss \"functions\", this package implements them as immutable types instead of true Julia functions. There are good reasons for that. For example it allows us to specify the properties of losse functions explicitly (e.g. isconvex(myloss)). It also makes for a more consistent API when it comes to computing the value or the derivative. Some loss functions even have additional parameters that need to be specified, such as the epsilon in the case of the epsilon-insensitive loss. Here, types allow for member variables to hide that information away from the method signatures.In order to avoid potential confusions with true Julia functions, we will refer to \"loss functions\" as \"losses\" instead. The available losses share a common interface for the most part. This section will provide an overview of the basic functionality that is available for all the different types of losses. We will discuss how to create a loss, how to compute its value and derivative, and how to query its properties."
},

{
    "location": "user/interface/#Instantiating-a-Loss-1",
    "page": "Working with Losses",
    "title": "Instantiating a Loss",
    "category": "section",
    "text": "Losses are immutable types. As such, one has to instantiate one in order to work with it. For most losses, the constructors do not expect any parameters.julia> L2DistLoss()\nLPDistLoss{2}()\n\njulia> HingeLoss()\nL1HingeLoss()We just said that we need to instantiate a loss in order to work with it. One could be inclined to belief, that it would be more memory-efficient to \"pre-allocate\" a loss when using it in more than one place.julia> loss = L2DistLoss()\nLPDistLoss{2}()\n\njulia> value(loss, 2, 3)\n1However, that is a common oversimplification. Because all losses are immutable types, they can live on the stack and thus do not come with a heap-allocation overhead.Even more interesting in the example above, is that for such losses as L2DistLoss, which do not have any constructor parameters or member variables, there is no additional code executed at all. Such singletons are only used for dispatch and don\'t even produce any additional code, which you can observe for yourself in the code below. As such they are zero-cost abstractions.julia> v1(loss,t,y) = value(loss,t,y)\n\njulia> v2(t,y) = value(L2DistLoss(),t,y)\n\njulia> @code_llvm v1(loss, 2, 3)\ndefine i64 @julia_v1_70944(i64, i64) #0 {\ntop:\n  %2 = sub i64 %1, %0\n  %3 = mul i64 %2, %2\n  ret i64 %3\n}\n\njulia> @code_llvm v2(2, 3)\ndefine i64 @julia_v2_70949(i64, i64) #0 {\ntop:\n  %2 = sub i64 %1, %0\n  %3 = mul i64 %2, %2\n  ret i64 %3\n}On the other hand, some types of losses are actually more comparable to whole families of losses instead of just a single one. For example, the immutable type L1EpsilonInsLoss has a free parameter epsilon. Each concrete epsilon results in a different concrete loss of the same family of epsilon-insensitive losses.julia> L1EpsilonInsLoss(0.5)\nL1EpsilonInsLoss{Float64}(0.5)\n\njulia> L1EpsilonInsLoss(1)\nL1EpsilonInsLoss{Float64}(1.0)For such losses that do have parameters, it can make a slight difference to pre-instantiate a loss. While they will live on the stack, the constructor usually performs some assertions and conversion for the given parameter. This can come at a slight overhead. At the very least it will not produce the same exact code when pre-instantiated. Still, the fact that they are immutable makes them very efficient abstractions with little to no performance overhead, and zero memory allocations on the heap."
},

{
    "location": "user/interface/#LearnBase.value-Tuple{SupervisedLoss,Number,Number}",
    "page": "Working with Losses",
    "title": "LearnBase.value",
    "category": "method",
    "text": "value(loss, target::Number, output::Number) -> Number\n\nCompute the (non-negative) numeric result for the loss-function denoted by the parameter loss and return it. Note that target and output can be of different numeric type, in which case promotion is performed in the manner appropriate for the given loss.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nL  Y times mathbbR rightarrow 0infty)\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we want to compute the value with.\ntarget::Number: The ground truth y in Y of the observation.\noutput::Number: The predicted output haty in mathbbR for the observation.\n\nExamples\n\n#               loss        y    ŷ\njulia> value(L1DistLoss(), 1.0, 2.0)\n1.0\n\njulia> value(L1DistLoss(), 1, 2)\n1\n\njulia> value(L1HingeLoss(), -1, 2)\n3\n\njulia> value(L1HingeLoss(), -1f0, 2f0)\n3.0f0\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.value-Tuple{SupervisedLoss,AbstractArray,AbstractArray}",
    "page": "Working with Losses",
    "title": "LearnBase.value",
    "category": "method",
    "text": "value(loss, targets::AbstractArray, outputs::AbstractArray)\n\nCompute the result of the loss function for each index-pair in targets and outputs individually and return the result as an array of the appropriate size.\n\nIn the case that the two parameters are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\n\nExamples\n\njulia> value(L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])\n3-element Array{Float64,1}:\n  1.0\n  9.0\n 25.0\n\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.value!-Tuple{AbstractArray,SupervisedLoss,AbstractArray,AbstractArray}",
    "page": "Working with Losses",
    "title": "LearnBase.value!",
    "category": "method",
    "text": "value!(buffer::AbstractArray, loss, targets::AbstractArray, outputs::AbstractArray) -> buffer\n\nCompute the result of the loss function for each index-pair in targets and outputs individually, and store them in the preallocated buffer. Note that buffer has to be of the appropriate size.\n\nIn the case that the two parameters, targets and outputs, are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nbuffer::AbstractArray: Array to store the computed values in. Old values will be overwritten and lost.\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\n\nExamples\n\njulia> buffer = zeros(3); # preallocate a buffer\n\njulia> value!(buffer, L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])\n3-element Array{Float64,1}:\n  1.0\n  9.0\n 25.0\n\n\n\n\n\n\n"
},

{
    "location": "user/interface/#Computing-the-Values-1",
    "page": "Working with Losses",
    "title": "Computing the Values",
    "category": "section",
    "text": "The first thing we may want to do is compute the loss for some observation (singular). In fact, all losses are implemented on single observations under the hood. The core function to compute the value of a loss is value. We will see throughout the documentation that this function allows for a lot of different method signatures to accomplish a variety of tasks.value(::SupervisedLoss, ::Number, ::Number)It may be interesting to note, that this function also supports broadcasting and all the syntax benefits that come with it. Thus, it is quite simple to make use of preallocated memory for storing the element-wise results.julia> value.(L1DistLoss(), [1,2,3], [2,5,-2])\n3-element Array{Int64,1}:\n 1\n 3\n 5\n\njulia> buffer = zeros(3); # preallocate a buffer\n\njulia> buffer .= value.(L1DistLoss(), [1.,2,3], [2,5,-2])\n3-element Array{Float64,1}:\n 1.0\n 3.0\n 5.0Furthermore, with the loop fusion changes that were introduced in Julia 0.6, one can also easily weight the influence of each observation without allocating a temporary array.julia> buffer .= value.(L1DistLoss(), [1.,2,3], [2,5,-2]) .* [2,1,0.5]\n3-element Array{Float64,1}:\n 2.0\n 3.0\n 2.5Even though broadcasting is supported, we do expose a vectorized method natively. This is done mainly for API consistency reasons. Internally it even uses broadcast itself, but it does provide the additional benefit of a more reliable type-inference.value(::SupervisedLoss, ::AbstractArray, ::AbstractArray)We also provide a mutating version for the same reasons. It even utilizes broadcast! underneath.value!(::AbstractArray, ::SupervisedLoss, ::AbstractArray, ::AbstractArray)"
},

{
    "location": "user/interface/#LearnBase.deriv-Tuple{SupervisedLoss,Number,Number}",
    "page": "Working with Losses",
    "title": "LearnBase.deriv",
    "category": "method",
    "text": "deriv(loss, target::Number, output::Number) -> Number\n\nCompute the derivative for the loss-function (denoted by the parameter loss) in respect to the output. Note that target and output can be of different numeric type, in which case promotion is performed in the manner appropriate for the given loss.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we want to compute the derivative with.\ntarget::Number: The ground truth y in Y of the observation.\noutput::Number: The predicted output haty in mathbbR for the observation.\n\nExamples\n\n#               loss        y    ŷ\njulia> deriv(L2DistLoss(), 1.0, 2.0)\n2.0\n\njulia> deriv(L2DistLoss(), 1, 2)\n2\n\njulia> deriv(L2HingeLoss(), -1, 2)\n6\n\njulia> deriv(L2HingeLoss(), -1f0, 2f0)\n6.0f0\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.deriv-Tuple{SupervisedLoss,AbstractArray,AbstractArray}",
    "page": "Working with Losses",
    "title": "LearnBase.deriv",
    "category": "method",
    "text": "deriv(loss, targets::AbstractArray, outputs::AbstractArray)\n\nCompute the derivative of the loss function in respect to the output for each index-pair in targets and outputs individually and return the result as an array of the appropriate size.\n\nIn the case that the two parameters are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\n\nExamples\n\njulia> deriv(L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])\n3-element Array{Float64,1}:\n   2.0\n   6.0\n -10.0\n\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.deriv!-Tuple{AbstractArray,SupervisedLoss,AbstractArray,AbstractArray}",
    "page": "Working with Losses",
    "title": "LearnBase.deriv!",
    "category": "method",
    "text": "deriv!(buffer::AbstractArray, loss, targets::AbstractArray, outputs::AbstractArray) -> buffer\n\nCompute the derivative of the loss function in respect to the output for each index-pair in targets and outputs individually, and store them in the preallocated buffer. Note that buffer has to be of the appropriate size.\n\nIn the case that the two parameters, targets and outputs, are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nbuffer::AbstractArray: Array to store the computed values in. Old values will be overwritten and lost.\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\n\nExamples\n\njulia> buffer = zeros(3); # preallocate a buffer\n\njulia> deriv!(buffer, L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])\n3-element Array{Float64,1}:\n   2.0\n   6.0\n -10.0\n\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.value_deriv-Tuple{SupervisedLoss,Number,Number}",
    "page": "Working with Losses",
    "title": "LearnBase.value_deriv",
    "category": "method",
    "text": "value_deriv(loss, target::Number, output::Number) -> Tuple\n\nReturn the results of value and deriv as a tuple, in which the first element is the value and the second element the derivative.\n\nIn some cases this function can yield better performance, because the losses can make use of shared variables when computing the results. Note that target and output can be of different numeric type, in which case promotion is performed in the manner appropriate for the given loss.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntarget::Number: The ground truth y in Y of the observation.\noutput::Number: The predicted output haty in mathbbR\n\nExamples\n\n#                     loss         y    ŷ\njulia> value_deriv(L2DistLoss(), -1.0, 3.0)\n(16.0, 8.0)\n\n\n\n"
},

{
    "location": "user/interface/#Computing-the-1st-Derivatives-1",
    "page": "Working with Losses",
    "title": "Computing the 1st Derivatives",
    "category": "section",
    "text": "Maybe the more interesting aspect of loss functions are their derivatives. In fact, most of the popular learning algorithm in Supervised Learning, such as gradient descent, utilize the derivatives of the loss in one way or the other during the training process.To compute the derivative of some loss we expose the function deriv. It supports the same exact method signatures as value. It may be interesting to note explicitly, that we always compute the derivative in respect to the predicted output, since we are interested in deducing in which direction the output should change.deriv(::SupervisedLoss, ::Number, ::Number)Similar to value, this function also supports broadcasting and all the syntax benefits that come with it. Thus, one can make use of preallocated memory for storing the element-wise derivatives.julia> deriv.(L2DistLoss(), [1,2,3], [2,5,-2])\n3-element Array{Int64,1}:\n   2\n   6\n -10\n\njulia> buffer = zeros(3); # preallocate a buffer\n\njulia> buffer .= deriv.(L2DistLoss(), [1.,2,3], [2,5,-2])\n3-element Array{Float64,1}:\n   2.0\n   6.0\n -10.0Furthermore, with the loop fusion changes that were introduced in Julia 0.6, one can also easily weight the influence of each observation without allocating a temporary array.julia> buffer .= deriv.(L2DistLoss(), [1.,2,3], [2,5,-2]) .* [2,1,0.5]\n3-element Array{Float64,1}:\n  4.0\n  6.0\n -5.0While broadcast is supported, we do expose a vectorized method natively. This is done mainly for API consistency reasons. Internally it even uses broadcast itself, but it does provide the additional benefit of a more reliable type-inference.deriv(::SupervisedLoss, ::AbstractArray, ::AbstractArray)We also provide a mutating version for the same reasons. It even utilizes broadcast underneath.deriv!(::AbstractArray, ::SupervisedLoss, ::AbstractArray, ::AbstractArray)It is also possible to compute the value and derivative at the same time. For some losses that means less computation overhead.value_deriv(::SupervisedLoss, ::Number, ::Number)"
},

{
    "location": "user/interface/#LearnBase.deriv2-Tuple{SupervisedLoss,Number,Number}",
    "page": "Working with Losses",
    "title": "LearnBase.deriv2",
    "category": "method",
    "text": "deriv2(loss, target::Number, output::Number) -> Number\n\nCompute the second derivative for the loss-function (denoted by the parameter loss) in respect to the output. Note that target and output can be of different numeric type, in which case promotion is performed in the manner appropriate for the given loss.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we want to compute the second derivative with.\ntarget::Number: The ground truth y in Y of the observation.\noutput::Number: The predicted output haty in mathbbR for the observation.\n\nExamples\n\n#               loss             y    ŷ\njulia> deriv2(LogitDistLoss(), -0.5, 0.3)\n0.42781939304058886\n\njulia> deriv2(LogitMarginLoss(), -1f0, 2f0)\n0.104993574f0\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.deriv2-Tuple{SupervisedLoss,AbstractArray,AbstractArray}",
    "page": "Working with Losses",
    "title": "LearnBase.deriv2",
    "category": "method",
    "text": "deriv2(loss, targets::AbstractArray, outputs::AbstractArray)\n\nCompute the second derivative of the loss function in respect to the output for each index-pair in targets and outputs individually and return the result as an array of the appropriate size.\n\nIn the case that the two parameters are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\n\nExamples\n\njulia> deriv2(L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])\n3-element Array{Float64,1}:\n 2.0\n 2.0\n 2.0\n\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LossFunctions.deriv2!-Tuple{AbstractArray,SupervisedLoss,AbstractArray,AbstractArray}",
    "page": "Working with Losses",
    "title": "LossFunctions.deriv2!",
    "category": "method",
    "text": "deriv2!(buffer::AbstractArray, loss, targets::AbstractArray, outputs::AbstractArray) -> buffer\n\nCompute the second derivative of the loss function in respect to the output for each index-pair in targets and outputs individually, and store them in the preallocated buffer. Note that buffer has to be of the appropriate size.\n\nIn the case that the two parameters, targets and outputs, are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nbuffer::AbstractArray: Array to store the computed values in. Old values will be overwritten and lost.\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\n\nExamples\n\njulia> buffer = zeros(3); # preallocate a buffer\n\njulia> deriv2!(buffer, L2DistLoss(), [1.0, 2.0, 3.0], [2, 5, -2])\n3-element Array{Float64,1}:\n 2.0\n 2.0\n 2.0\n\n\n\n\n\n\n"
},

{
    "location": "user/interface/#Computing-the-2nd-Derivatives-1",
    "page": "Working with Losses",
    "title": "Computing the 2nd Derivatives",
    "category": "section",
    "text": "Additionally to the first derivative, we also provide the corresponding methods for the second derivative through the function deriv2. Note again, that we always compute the derivative in respect to the predicted output.deriv2(::SupervisedLoss, ::Number, ::Number)Just like deriv and value, this function also supports broadcasting and all the syntax benefits that come with it. Thus, one can make use of preallocated memory for storing the element-wise derivatives.julia> deriv2.(LogitDistLoss(), [-0.5, 1.2, 3], [0.3, 2.3, -2])\n3-element Array{Float64,1}:\n 0.42781939304058886\n 0.3747397590950412\n 0.013296113341580313\n\njulia> buffer = zeros(3); # preallocate a buffer\n\njulia> buffer .= deriv2.(LogitDistLoss(), [-0.5, 1.2, 3], [0.3, 2.3, -2])\n3-element Array{Float64,1}:\n 0.42781939304058886\n 0.3747397590950412\n 0.013296113341580313Furthermore deriv2 supports all the same method signatures as deriv does.deriv2(::SupervisedLoss, ::AbstractArray, ::AbstractArray)\nderiv2!(::AbstractArray, ::SupervisedLoss, ::AbstractArray, ::AbstractArray)"
},

{
    "location": "user/interface/#LossFunctions.value_fun-Tuple{SupervisedLoss}",
    "page": "Working with Losses",
    "title": "LossFunctions.value_fun",
    "category": "method",
    "text": "value_fun(loss::SupervisedLoss) -> Function\n\nReturns a new function that computes the value for the given loss. This new function will support all the signatures that value does.\n\njulia> f = value_fun(L2DistLoss());\n\njulia> f(-1.0, 3.0) # computes the value of L2DistLoss\n16.0\n\njulia> f.([1.,2], [4,7])\n2-element Array{Float64,1}:\n  9.0\n 25.0\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LossFunctions.deriv_fun-Tuple{SupervisedLoss}",
    "page": "Working with Losses",
    "title": "LossFunctions.deriv_fun",
    "category": "method",
    "text": "deriv_fun(loss::SupervisedLoss) -> Function\n\nReturns a new function that computes the deriv for the given loss. This new function will support all the signatures that deriv does.\n\njulia> g = deriv_fun(L2DistLoss());\n\njulia> g(-1.0, 3.0) # computes the deriv of L2DistLoss\n8.0\n\njulia> g.([1.,2], [4,7])\n2-element Array{Float64,1}:\n  6.0\n 10.0\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LossFunctions.deriv2_fun-Tuple{SupervisedLoss}",
    "page": "Working with Losses",
    "title": "LossFunctions.deriv2_fun",
    "category": "method",
    "text": "deriv2_fun(loss::SupervisedLoss) -> Function\n\nReturns a new function that computes the deriv2 (i.e. second derivative) for the given loss. This new function will support all the signatures that deriv2 does.\n\njulia> g2 = deriv2_fun(L2DistLoss());\n\njulia> g2(-1.0, 3.0) # computes the second derivative of L2DistLoss\n2.0\n\njulia> g2.([1.,2], [4,7])\n2-element Array{Float64,1}:\n 2.0\n 2.0\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LossFunctions.value_deriv_fun-Tuple{SupervisedLoss}",
    "page": "Working with Losses",
    "title": "LossFunctions.value_deriv_fun",
    "category": "method",
    "text": "value_deriv_fun(loss::SupervisedLoss) -> Function\n\nReturns a new function that computes the value_deriv for the given loss. This new function will support all the signatures that value_deriv does.\n\njulia> fg = value_deriv_fun(L2DistLoss());\n\njulia> fg(-1.0, 3.0) # computes the second derivative of L2DistLoss\n(16.0, 8.0)\n\n\n\n\n\n"
},

{
    "location": "user/interface/#Function-Closures-1",
    "page": "Working with Losses",
    "title": "Function Closures",
    "category": "section",
    "text": "In some circumstances it may be convenient to have the loss function or its derivative as a proper Julia function. Instead of exporting special function names for every implemented loss (like l2distloss(...)), we provide the ability to generate a true function on the fly for any given loss.value_fun(::SupervisedLoss)\nderiv_fun(::SupervisedLoss)\nderiv2_fun(::SupervisedLoss)\nvalue_deriv_fun(::SupervisedLoss)"
},

{
    "location": "user/interface/#LearnBase.isconvex",
    "page": "Working with Losses",
    "title": "LearnBase.isconvex",
    "category": "function",
    "text": "isconvex(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss denotes a convex function. A function f  mathbbR^n rightarrow mathbbR is convex if its domain is a convex set and if for all x y in that domain, with theta such that for 0 leq theta leq 1, we have\n\nf(theta x + (1 - theta) y) leq theta f(x) + (1 - theta) f(y)\n\nExamples\n\njulia> isconvex(LPDistLoss(0.5))\nfalse\n\njulia> isconvex(ZeroOneLoss())\nfalse\n\njulia> isconvex(L1DistLoss())\ntrue\n\njulia> isconvex(L2DistLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isstrictlyconvex",
    "page": "Working with Losses",
    "title": "LearnBase.isstrictlyconvex",
    "category": "function",
    "text": "isstrictlyconvex(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss denotes a strictly convex function. A function f  mathbbR^n rightarrow mathbbR is strictly convex if its domain is a convex set and if for all x y in that domain where x neq y, with theta such that for 0  theta  1, we have\n\nf(theta x + (1 - theta) y)  theta f(x) + (1 - theta) f(y)\n\nExamples\n\njulia> isstrictlyconvex(L1DistLoss())\nfalse\n\njulia> isstrictlyconvex(LogitDistLoss())\ntrue\n\njulia> isstrictlyconvex(L2DistLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isstronglyconvex",
    "page": "Working with Losses",
    "title": "LearnBase.isstronglyconvex",
    "category": "function",
    "text": "isstronglyconvex(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss denotes a strongly convex function. A function f  mathbbR^n rightarrow mathbbR is m-strongly convex if its domain is a convex set and if forall xy in dom f where x neq y, and theta such that for 0 le theta le 1 , we have\n\nf(theta x + (1 - theta)y)  theta f(x) + (1 - theta) f(y) - 05 m cdot theta (1 - theta)  x - y _2^2\n\nIn a more familiar setting, if the loss function is differentiable we have\n\nleft( nabla f(x) - nabla f(y) right)^top (x - y) ge m  x - y_2^2\n\nExamples\n\njulia> isstronglyconvex(L1DistLoss())\nfalse\n\njulia> isstronglyconvex(LogitDistLoss())\nfalse\n\njulia> isstronglyconvex(L2DistLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isdifferentiable",
    "page": "Working with Losses",
    "title": "LearnBase.isdifferentiable",
    "category": "function",
    "text": "isdifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool\n\nReturn true if the given loss is differentiable (optionally limited to the given point x if specified).\n\nA function f  mathbbR^n rightarrow mathbbR^m is differentiable at a point x in int dom f, if there exists a matrix Df(x) in mathbbR^m times n such that it satisfies:\n\nlim_z neq x z to x fracf(z) - f(x) - Df(x)(z-x)_2z - x_2 = 0\n\nA function is differentiable if its domain is open and it is differentiable at every point x.\n\nExamples\n\njulia> isdifferentiable(L1DistLoss())\nfalse\n\njulia> isdifferentiable(L1DistLoss(), 1)\ntrue\n\njulia> isdifferentiable(L2DistLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.istwicedifferentiable",
    "page": "Working with Losses",
    "title": "LearnBase.istwicedifferentiable",
    "category": "function",
    "text": "istwicedifferentiable(loss::SupervisedLoss, [x::Number]) -> Bool\n\nReturn true if the given loss is differentiable (optionally limited to the given point x if specified).\n\nA function f  mathbbR^n rightarrow mathbbR is said to be twice differentiable at a point x in int dom f, if the function derivative for nabla f exists at x.\n\nnabla^2 f(x) = D nabla f(x)\n\nA function is twice differentiable if its domain is open and it is twice differentiable at every point x.\n\nExamples\n\njulia> isdifferentiable(L1DistLoss())\nfalse\n\njulia> isdifferentiable(L1DistLoss(), 1)\ntrue\n\njulia> isdifferentiable(L2DistLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.islocallylipschitzcont",
    "page": "Working with Losses",
    "title": "LearnBase.islocallylipschitzcont",
    "category": "function",
    "text": "islocallylipschitzcont(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss function is locally-Lipschitz continous.\n\nA supervised loss L  Y times mathbbR rightarrow 0 infty) is called locally Lipschitz continuous if forall a ge 0 there exists a constant :math:c_a \\ge 0, such that\n\nsup_y in Y left L(yt)  L(yt) right le c_a t  t  qquad  tt in aa\n\nEvery convex function is locally lipschitz continuous\n\nExamples\n\njulia> islocallylipschitzcont(ExpLoss())\ntrue\n\njulia> islocallylipschitzcont(SigmoidLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.islipschitzcont",
    "page": "Working with Losses",
    "title": "LearnBase.islipschitzcont",
    "category": "function",
    "text": "islipschitzcont(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss function is Lipschitz continuous.\n\nA supervised loss function L  Y times mathbbR rightarrow 0 infty) is Lipschitz continous, if there exists a finite constant M  infty such that\n\nL(y t) - L(y t) le M t - t  qquad  forall (y t) in Y times mathbbR\n\nExamples\n\njulia> islipschitzcont(SigmoidLoss())\ntrue\n\njulia> islipschitzcont(ExpLoss())\nfalse\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isnemitski",
    "page": "Working with Losses",
    "title": "LearnBase.isnemitski",
    "category": "function",
    "text": "isnemitski(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss denotes a Nemitski loss function.\n\nWe call a supervised loss function L  Y times mathbbR rightarrow 0infty) a Nemitski loss if there exist a measurable function b  Y rightarrow 0 infty) and an increasing function h  0 infty) rightarrow 0 infty) such that\n\nL(yhaty) le b(y) + h(haty)  qquad  (y haty) in Y times mathbbR\n\nIf a loss if locally lipsschitz continuous then it is a Nemitski loss\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isclipable",
    "page": "Working with Losses",
    "title": "LearnBase.isclipable",
    "category": "function",
    "text": "isclipable(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss function is clipable. A supervised loss L  Y times mathbbR rightarrow 0 infty) can be clipped at M  0 if, for all (yt) in Y times mathbbR,\n\nL(y hatt) le L(y t)\n\nwhere hatt denotes the clipped value of t at pm M. That is\n\nhatt = begincases -M  quad textif  t  -M  t  quad textif  t in -M M  M  quad textif  t  M endcases\n\nExamples\n\njulia> isclipable(ExpLoss())\nfalse\n\njulia> isclipable(L2DistLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.ismarginbased",
    "page": "Working with Losses",
    "title": "LearnBase.ismarginbased",
    "category": "function",
    "text": "ismarginbased(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss is a margin-based loss.\n\nA supervised loss function L  Y times mathbbR rightarrow 0 infty) is said to be margin-based, if there exists a representing function psi  mathbbR rightarrow 0 infty) satisfying\n\nL(y haty) = psi (y cdot haty)  qquad  (y haty) in Y times mathbbR\n\nExamples\n\njulia> ismarginbased(HuberLoss(2))\nfalse\n\njulia> ismarginbased(L2MarginLoss())\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isclasscalibrated",
    "page": "Working with Losses",
    "title": "LearnBase.isclasscalibrated",
    "category": "function",
    "text": "isclasscalibrated(loss::SupervisedLoss) -> Bool\n\n\n\n\n\n"
},

{
    "location": "user/interface/#LearnBase.isdistancebased",
    "page": "Working with Losses",
    "title": "LearnBase.isdistancebased",
    "category": "function",
    "text": "isdistancebased(loss::SupervisedLoss) -> Bool\n\nReturn true ifthe given loss is a distance-based loss.\n\nA supervised loss function L  Y times mathbbR rightarrow 0 infty) is said to be distance-based, if there exists a representing function psi  mathbbR rightarrow 0 infty) satisfying psi (0) = 0 and\n\nL(y haty) = psi (haty - y)  qquad  (y haty) in Y times mathbbR\n\nExamples\n\njulia> isdistancebased(HuberLoss(2))\ntrue\n\njulia> isdistancebased(L2MarginLoss())\nfalse\n\n\n\n"
},

{
    "location": "user/interface/#LinearAlgebra.issymmetric",
    "page": "Working with Losses",
    "title": "LinearAlgebra.issymmetric",
    "category": "function",
    "text": "issymmetric(loss::SupervisedLoss) -> Bool\n\nReturn true if the given loss is a symmetric loss.\n\nA function f  mathbbR rightarrow 0infty) is said to be symmetric about origin if we have\n\nf(x) = f(-x) qquad  forall x in mathbbR\n\nA distance-based loss is said to be symmetric if its representing function is symmetric.\n\nExamples\n\njulia> issymmetric(QuantileLoss(0.2))\nfalse\n\njulia> issymmetric(LPDistLoss(2))\ntrue\n\n\n\n"
},

{
    "location": "user/interface/#Properties-of-a-Loss-1",
    "page": "Working with Losses",
    "title": "Properties of a Loss",
    "category": "section",
    "text": "In some situations it can be quite useful to assert certain properties about a loss-function. One such scenario could be when implementing an algorithm that requires the loss to be strictly convex or Lipschitz continuous. Note that we will only skim over the defintions in most cases. A good treatment of all of the concepts involved can be found in either [BOYD2004] or [STEINWART2008].[BOYD2004]: Stephen Boyd and Lieven Vandenberghe. \"Convex Optimization\". Cambridge University Press, 2004.[STEINWART2008]: Steinwart, Ingo, and Andreas Christmann. \"Support vector machines\". Springer Science & Business Media, 2008.This package uses functions to represent individual properties of a loss. It follows a list of implemented property-functions defined in LearnBase.jl.isconvex\nisstrictlyconvex\nisstronglyconvex\nisdifferentiable\nistwicedifferentiable\nislocallylipschitzcont\nislipschitzcont\nisnemitski\nisclipable\nismarginbased\nisclasscalibrated\nisdistancebased\nissymmetric"
},

{
    "location": "user/aggregate/#",
    "page": "Efficient Sum and Mean",
    "title": "Efficient Sum and Mean",
    "category": "page",
    "text": "DocTestSetup = quote\n    using LossFunctions\nend"
},

{
    "location": "user/aggregate/#Efficient-Sum-and-Mean-1",
    "page": "Efficient Sum and Mean",
    "title": "Efficient Sum and Mean",
    "category": "section",
    "text": "In many situations we are not really that interested in the individual loss values (or derivatives) of each observation, but the sum or mean of them; be it weighted or unweighted. For example, by computing the unweighted mean of the loss for our training set, we would effectively compute what is known as the empirical risk. This is usually the quantity (or an important part of it) that we are interesting in minimizing.When we say \"weighted\" or \"unweighted\", we are referring to whether we are explicitly specifying the influence of individual observations on the result. \"Weighing\" an observation is achieved by multiplying its value with some number (i.e. the \"weight\" of that observation). As a consequence that weighted observation will have a stronger or weaker influence on the result. In order to weigh an observation we have to know which array dimension (if there are more than one) denotes the observations. On the other hand, for computing an unweighted result we don\'t actually need to know anything about the meaning of the array dimensions, as long as the targets and the outputs are of compatible shape and size.The naive way to compute such an unweighted reduction, would be to call mean or sum on the result of the element-wise operation. The following code snipped show an example of that. We say \"naive\", because it will not give us an acceptable performance.julia> value(L1DistLoss(), [1.,2,3], [2,5,-2])\n3-element Array{Float64,1}:\n 1.0\n 3.0\n 5.0\n\njulia> sum(value(L1DistLoss(), [1.,2,3], [2,5,-2])) # WARNING: Bad code\n9.0This works as expected, but there is a price for it. Before the sum can be computed, value will allocate a temporary array and fill it with the element-wise results. After that, sum will iterate over this temporary array and accumulate the values accordingly. Bottom line: we allocate temporary memory that we don\'t need in the end and could avoid.For that reason we provide special methods that compute the common accumulations efficiently without allocating temporary arrays. These methods can be invoked using an additional parameter which specifies how the values should be accumulated / averaged. The type of this parameter has to be a subtype of AverageMode."
},

{
    "location": "user/aggregate/#Average-Modes-1",
    "page": "Efficient Sum and Mean",
    "title": "Average Modes",
    "category": "section",
    "text": "Before we discuss these memory-efficient methods, let us briefly introduce the available average mode types. We provide a number of different averages modes, all of which are contained within the namespace AvgMode. An instance of such type can then be used as additional parameter to value, deriv, and deriv2, as we will see further down.It follows a list of available average modes. Each of which with a short description of what their effect would be when used as an additional parameter to the functions mentioned above.AvgMode.None\nAvgMode.Sum\nAvgMode.Mean\nAvgMode.WeightedSum\nAvgMode.WeightedMean"
},

{
    "location": "user/aggregate/#LearnBase.value-Tuple{Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.value",
    "category": "method",
    "text": "value(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) -> Number\n\nCompute the weighted or unweighted sum or mean (depending on avgmode) of the individual values of the loss function for each pair in targets and outputs. This method will not allocate a temporary array.\n\nIn the case that the two parameters are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\n\nExamples\n\njulia> value(L1DistLoss(), [1,2,3], [2,5,-2], AggMode.Sum())\n9\n\njulia> value(L1DistLoss(), [1.,2,3], [2,5,-2], AggMode.Sum())\n9.0\n\njulia> value(L1DistLoss(), [1,2,3], [2,5,-2], AggMode.Mean())\n3.0\n\njulia> value(L1DistLoss(), Float32[1,2,3], Float32[2,5,-2], AggMode.Mean())\n3.0f0\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LearnBase.deriv-Tuple{Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.deriv",
    "category": "method",
    "text": "deriv(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) -> Number\n\nCompute the weighted or unweighted sum or mean (depending on avgmode) of the individual derivatives of the loss function for each pair in targets and outputs. This method will not allocate a temporary array.\n\nIn the case that the two parameters are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\n\nExamples\n\njulia> deriv(L2DistLoss(), [1,2,3], [2,5,-2], AggMode.Sum())\n-2\n\njulia> deriv(L2DistLoss(), [1,2,3], [2,5,-2], AggMode.Mean())\n-0.6666666666666666\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LearnBase.deriv2-Tuple{Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.deriv2",
    "category": "method",
    "text": "deriv2(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode) -> Number\n\nCompute the weighted or unweighted sum or mean (depending on avgmode) of the individual second derivatives of the loss function for each pair in targets and outputs. This method will not allocate a temporary array.\n\nIn the case that the two parameters are arrays with a different number of dimensions, broadcast will be performed. Note that the given parameters are expected to have the same size in the dimensions they share.\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\n\nExamples\n\njulia> deriv2(LogitDistLoss(), [1.,2,3], [2,5,-2], AggMode.Sum())\n0.49687329928636825\n\njulia> deriv2(LogitDistLoss(), [1.,2,3], [2,5,-2], AggMode.Mean())\n0.1656244330954561\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#Unweighted-Sum-and-Mean-1",
    "page": "Efficient Sum and Mean",
    "title": "Unweighted Sum and Mean",
    "category": "section",
    "text": "As hinted before, we provide special memory efficient methods for computing the sum or the mean of the element-wise (or broadcasted) results of value, deriv, and deriv2. These methods avoid the allocation of a temporary array and instead compute the result directly.value(::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode)The exact same method signature is also implemented for deriv and deriv2 respectively.deriv(::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode)\nderiv2(::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode)"
},

{
    "location": "user/aggregate/#LearnBase.value-Tuple{Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode,LearnBase.ObsDimension}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.value",
    "category": "method",
    "text": "value(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> AbstractVector\n\nCompute the values of the loss function for each pair in targets and outputs individually, and return either the weighted or unweighted sum or mean for each observation (depending on avgmode). This method will not allocate a temporary array, but it will allocate the resulting vector.\n\nBoth arrays have to be of the same shape and size. Furthermore they have to have at least two array dimensions (i.e. they must not be vectors).\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\nobsdim::ObsDimension: Specifies which of the array dimensions denotes the observations. see ?ObsDim for more information.\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LearnBase.value!-Tuple{AbstractArray,Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode,LearnBase.ObsDimension}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.value!",
    "category": "method",
    "text": "value!(buffer::AbstractArray, loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> buffer\n\nCompute the values of the loss function for each pair in targets and outputs individually, and return either the weighted or unweighted sum or mean for each observation, depending on avgmode. The results are stored into the given vector buffer. This method will not allocate a temporary array.\n\nBoth arrays have to be of the same shape and size. Furthermore they have to have at least two array dimensions (i.e. so they must not be vectors).\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nbuffer::AbstractArray: Array to store the computed values in. Old values will be overwritten and lost.\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\nobsdim::ObsDimension: Specifies which of the array dimensions denotes the observations. see ?ObsDim for more information.\n\nExamples\n\njulia> targets = reshape(1:8, (2, 4)) ./ 8;\n\njulia> outputs = reshape(1:2:16, (2, 4)) ./ 8;\n\njulia> buffer = zeros(2);\n\njulia> value!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.First())\n2-element Array{Float64,1}:\n 1.5\n 2.0\n\njulia> buffer = zeros(4);\n\njulia> value!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.Last())\n4-element Array{Float64,1}:\n 0.125\n 0.625\n 1.125\n 1.625\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LearnBase.deriv-Tuple{Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode,LearnBase.ObsDimension}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.deriv",
    "category": "method",
    "text": "deriv(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> AbstractVector\n\nCompute the derivative of the loss function for each pair in targets and outputs individually, and return either the weighted or unweighted sum or mean for each observation (depending on avgmode). This method will not allocate a temporary array, but it will allocate the resulting vector.\n\nBoth arrays have to be of the same shape and size. Furthermore they have to have at least two array dimensions (i.e. they must not be vectors).\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\nobsdim::ObsDimension: Specifies which of the array dimensions denotes the observations. see ?ObsDim for more information.\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LearnBase.deriv!-Tuple{AbstractArray,Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode,LearnBase.ObsDimension}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.deriv!",
    "category": "method",
    "text": "deriv!(buffer::AbstractArray, loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> buffer\n\nCompute the derivative of the loss function for each pair in targets and outputs individually, and return either the weighted or unweighted sum or mean for each observation, depending on avgmode. The results are stored into the given vector buffer. This method will not allocate a temporary array.\n\nBoth arrays have to be of the same shape and size. Furthermore they have to have at least two array dimensions (i.e. so they must not be vectors).\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nbuffer::AbstractArray: Array to store the computed values in. Old values will be overwritten and lost.\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\nobsdim::ObsDimension: Specifies which of the array dimensions denotes the observations. see ?ObsDim for more information.\n\nExamples\n\njulia> targets = reshape(1:8, (2, 4)) ./ 8;\n\njulia> outputs = reshape(1:2:16, (2, 4)) ./ 8;\n\njulia> buffer = zeros(2);\n\njulia> deriv!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.First())\n2-element Array{Float64,1}:\n 3.0\n 4.0\n\njulia> buffer = zeros(4);\n\njulia> deriv!(buffer, L1DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.Last())\n4-element Array{Float64,1}:\n 1.0\n 2.0\n 2.0\n 2.0\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LearnBase.deriv2-Tuple{Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode,LearnBase.ObsDimension}",
    "page": "Efficient Sum and Mean",
    "title": "LearnBase.deriv2",
    "category": "method",
    "text": "deriv2(loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> AbstractVector\n\nCompute the second derivative of the loss function for each pair in targets and outputs individually, and return either the weighted or unweighted sum or mean for each observation (depending on avgmode). This method will not allocate a temporary array, but it will allocate the resulting vector.\n\nBoth arrays have to be of the same shape and size. Furthermore they have to have at least two array dimensions (i.e. they must not be vectors).\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\nobsdim::ObsDimension: Specifies which of the array dimensions denotes the observations. see ?ObsDim for more information.\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#LossFunctions.deriv2!-Tuple{AbstractArray,Loss,AbstractArray,AbstractArray,LossFunctions.AggregateMode,LearnBase.ObsDimension}",
    "page": "Efficient Sum and Mean",
    "title": "LossFunctions.deriv2!",
    "category": "method",
    "text": "deriv2!(buffer::AbstractArray, loss, target::AbstractArray, output::AbstractArray, avgmode::AggregateMode, obsdim::ObsDimension) -> buffer\n\nCompute the second derivative of the loss function for each pair in targets and outputs individually, and return either the weighted or unweighted sum or mean for each observation, depending on avgmode. The results are stored into the given vector buffer. This method will not allocate a temporary array.\n\nBoth arrays have to be of the same shape and size. Furthermore they have to have at least two array dimensions (i.e. so they must not be vectors).\n\nNote: This function should always be type-stable. If it isn\'t, you likely found a bug.\n\nArguments\n\nbuffer::AbstractArray: Array to store the computed values in. Old values will be overwritten and lost.\nloss::SupervisedLoss: The loss-function L we are working with.\ntargets::AbstractArray: The array of ground truths mathbfy.\noutputs::AbstractArray: The array of predicted outputs mathbfhaty.\navgmode::AggregateMode: Must be one of the following: AggMode.Sum(), AggMode.Mean(), AggMode.WeightedSum, or AggMode.WeightedMean.\nobsdim::ObsDimension: Specifies which of the array dimensions denotes the observations. see ?ObsDim for more information.\n\nExamples\n\njulia> targets = reshape(1:8, (2, 4)) ./ 8;\n\njulia> outputs = reshape(1:2:16, (2, 4)) ./ 8;\n\njulia> buffer = zeros(2);\n\njulia> deriv2!(buffer, L2DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.First())\n2-element Array{Float64,1}:\n 8.0\n 8.0\n\njulia> buffer = zeros(4);\n\njulia> deriv2!(buffer, L2DistLoss(), targets, outputs, AggMode.Sum(), ObsDim.Last())\n4-element Array{Float64,1}:\n 4.0\n 4.0\n 4.0\n 4.0\n\n\n\n\n\n"
},

{
    "location": "user/aggregate/#Sum-and-Mean-per-Observation-1",
    "page": "Efficient Sum and Mean",
    "title": "Sum and Mean per Observation",
    "category": "section",
    "text": "When the targets and predicted outputs are multi-dimensional arrays instead of vectors, we may be interested in accumulating the values over all but one dimension. This is typically the case when we work in a multi-variable regression setting, where each observation has multiple outputs and thus multiple targets. In those scenarios we may be more interested in the average loss for each observation, rather than the total average over all the data.To be able to accumulate the values for each observation separately, we have to know and explicitly specify the dimension that denotes the observations. For that purpose we provide the types contained in the namespace ObsDim.value(::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode, ::LearnBase.ObsDimension)Consider the following two matrices, targets and outputs. We will fill them with some generated example values in order to better understand the effects of later operations.julia> targets = reshape(1:8, (2, 4)) ./ 8\n2×4 Array{Float64,2}:\n 0.125  0.375  0.625  0.875\n 0.25   0.5    0.75   1.0\n\njulia> outputs = reshape(1:2:16, (2, 4)) ./ 8\n2×4 Array{Float64,2}:\n 0.125  0.625  1.125  1.625\n 0.375  0.875  1.375  1.875There are two ways to interpret the shape of these arrays if one dimension is supposed to denote the observations. The first interpretation would be to say that the first dimension denotes the observations. Thus this data would consist of two observations with four variables each.julia> value(L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.First())\n2-element Array{Float64,1}:\n 1.5\n 2.0\n\njulia> value(L1DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.First())\n2-element Array{Float64,1}:\n 0.375\n 0.5The second possible interpretation would be to say that the second/last dimension denotes the observations. In that case our data consists of four observations with two variables each.julia> value(L1DistLoss(), targets, outputs, AvgMode.Sum(), ObsDim.Last())\n4-element Array{Float64,1}:\n 0.125\n 0.625\n 1.125\n 1.625\n\njulia> value(L1DistLoss(), targets, outputs, AvgMode.Mean(), ObsDim.Last())\n4-element Array{Float64,1}:\n 0.0625\n 0.3125\n 0.5625\n 0.8125Because this method returns a vector of values, we also provide a mutating version that can make use a preallocated vector to write the results into.value!(::AbstractArray, ::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode, ::LearnBase.ObsDimension)Naturally we also provide both of these methods for deriv and deriv2 respectively.deriv(::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode, ::LearnBase.ObsDimension)\nderiv!(::AbstractArray, ::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode, ::LearnBase.ObsDimension)\nderiv2(::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode, ::LearnBase.ObsDimension)\nderiv2!(::AbstractArray, ::Loss, ::AbstractArray, ::AbstractArray, ::LossFunctions.AverageMode, ::LearnBase.ObsDimension)"
},

{
    "location": "user/aggregate/#Weighted-Sum-and-Mean-1",
    "page": "Efficient Sum and Mean",
    "title": "Weighted Sum and Mean",
    "category": "section",
    "text": "Up to this point, all the averaging was performed in an unweighted manner. That means that each observation was treated as equal and had thus the same potential influence on the result. In this sub-section we will consider the situations in which we do want to explicitly specify the influence of each observation (i.e. we want to weigh them). When we say we \"weigh\" an observation, what it effectively boils down to is multiplying the result for that observation (i.e. the computed loss or derivative) with some number. This is done for every observation individually.To get a better understand of what we are talking about, let us consider performing a weighting scheme manually. The following code will compute the loss for three observations, and then multiply the result of the second observation with the number 2, while the other two remains as they are. If we then sum up the results, we will see that the loss of the second observation was effectively counted twice.julia> result = value.(L1DistLoss(), [1.,2,3], [2,5,-2]) .* [1,2,1]\n3-element Array{Float64,1}:\n 1.0\n 6.0\n 5.0\n\njulia> sum(result)\n12.0The point of weighing observations is to inform the learning algorithm we are working with, that it is more important to us to predict some observations correctly than it is for others. So really, the concrete weight-factor matters less than the ratio between the different weights. In the example above the second observation was thus considered twice as important as any of the other two observations.In the case of multi-dimensional arrays the process isn\'t that simple anymore. In such a scenario, computing the weighted sum (or weighted mean) can be thought of as having an additional step. First we either compute the sum or (unweighted) average for each observation (which results in a vector), and then we compute the weighted sum of all observations.The following code snipped demonstrates how to compute the AvgMode.WeightedSum([2,1]) manually. This is not meant as an example of how to do it, but simply to show what is happening qualitatively. In this example we assume that we are working in a multi-variable regression setting, in which our data set has four observations with two target-variables each.julia> targets = reshape(1:8, (2, 4)) ./ 8\n2×4 Array{Float64,2}:\n 0.125  0.375  0.625  0.875\n 0.25   0.5    0.75   1.0\n\njulia> outputs = reshape(1:2:16, (2, 4)) ./ 8\n2×4 Array{Float64,2}:\n 0.125  0.625  1.125  1.625\n 0.375  0.875  1.375  1.875\n\njulia> # WARNING: BAD CODE - ONLY FOR ILLUSTRATION\n\njulia> tmp = sum(value.(L1DistLoss(), targets, outputs), dims=2) # assuming ObsDim.First()\n2×1 Array{Float64,2}:\n 1.5\n 2.0\n\njulia> sum(tmp .* [2, 1]) # weigh 1st observation twice as high\n5.0To manually compute the result for AvgMode.WeightedMean([2,1]) we follow a similar approach, but use the normalized weight vector in the last step.julia> using Statistics # for access to \"mean\"\n\njulia> # WARNING: BAD CODE - ONLY FOR ILLUSTRATION\n\njulia> tmp = mean(value.(L1DistLoss(), targets, outputs), dims=2) # ObsDim.First()\n2×1 Array{Float64,2}:\n 0.375\n 0.5\n\njulia> sum(tmp .* [0.6666, 0.3333]) # weigh 1st observation twice as high\n0.416625Note that you can specify explicitly if you want to normalize the weight vector. That option is supported for computing the weighted sum, as well as for computing the weighted mean. See the documentation for AvgMode.WeightedSum and AvgMode.WeightedMean for more information.The code-snippets above are of course very inefficient, because they allocate (multiple) temporary arrays. We only included them to demonstrate what is happening in terms of desired result / effect. For doing those computations efficiently we provide special methods for value, deriv, deriv2 and their mutating counterparts.julia> value(L1DistLoss(), [1.,2,3], [2,5,-2], AvgMode.WeightedSum([1,2,1]))\n12.0\n\njulia> value(L1DistLoss(), [1.,2,3], [2,5,-2], AvgMode.WeightedMean([1,2,1]))\n3.0\n\njulia> value(L1DistLoss(), targets, outputs, AvgMode.WeightedSum([2,1]), ObsDim.First())\n5.0\n\njulia> value(L1DistLoss(), targets, outputs, AvgMode.WeightedMean([2,1]), ObsDim.First())\n0.4166666666666667We also provide this functionality for deriv and deriv2 respectively.julia> deriv(L2DistLoss(), [1.,2,3], [2,5,-2], AvgMode.WeightedSum([1,2,1]))\n4.0\n\njulia> deriv(L2DistLoss(), [1.,2,3], [2,5,-2], AvgMode.WeightedMean([1,2,1]))\n1.0\n\njulia> deriv(L2DistLoss(), targets, outputs, AvgMode.WeightedSum([2,1]), ObsDim.First())\n10.0\n\njulia> deriv(L2DistLoss(), targets, outputs, AvgMode.WeightedMean([2,1]), ObsDim.First())\n0.8333333333333334"
},

{
    "location": "losses/distance/#",
    "page": "Distance-based Losses",
    "title": "Distance-based Losses",
    "category": "page",
    "text": "DocTestSetup = quote\n    using LossFunctions\nend<div class=\"loss-docs\">"
},

{
    "location": "losses/distance/#Distance-based-Losses-1",
    "page": "Distance-based Losses",
    "title": "Distance-based Losses",
    "category": "section",
    "text": "Loss functions that belong to the category \"distance-based\" are primarily used in regression problems. They utilize the numeric difference between the predicted output and the true target as a proxy variable to quantify the quality of individual predictions.This section lists all the subtypes of DistanceLoss that are implemented in this package."
},

{
    "location": "losses/distance/#LossFunctions.LPDistLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.LPDistLoss",
    "category": "type",
    "text": "LPDistLoss{P} <: DistanceLoss\n\nThe P-th power absolute distance loss. It is Lipschitz continuous iff P == 1, convex if and only if P >= 1, and strictly convex iff P > 1.\n\nL(r) = r^P\n\n\n\n"
},

{
    "location": "losses/distance/#LPDistLoss-1",
    "page": "Distance-based Losses",
    "title": "LPDistLoss",
    "category": "section",
    "text": "LPDistLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = mid r mid ^p L(r) = p cdot r cdot mid r mid ^p-2"
},

{
    "location": "losses/distance/#LossFunctions.L1DistLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.L1DistLoss",
    "category": "type",
    "text": "L1DistLoss <: DistanceLoss\n\nThe absolute distance loss. Special case of the LPDistLoss with P=1. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(r) = r\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    3 │\\.                     ./│    1 │            ┌------------│\n      │ \'\\.                 ./\' │      │            |            │\n      │   \\.               ./   │      │            |            │\n      │    \'\\.           ./\'    │      │_           |           _│\n    L │      \\.         ./      │   L\' │            |            │\n      │       \'\\.     ./\'       │      │            |            │\n      │         \\.   ./         │      │            |            │\n    0 │          \'\\./\'          │   -1 │------------┘            │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -3                        3      -3                        3\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#L1DistLoss-1",
    "page": "Distance-based Losses",
    "title": "L1DistLoss",
    "category": "section",
    "text": "L1DistLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = mid r mid L(r) = textrmsign(r)"
},

{
    "location": "losses/distance/#LossFunctions.L2DistLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.L2DistLoss",
    "category": "type",
    "text": "L2DistLoss <: DistanceLoss\n\nThe least squares loss. Special case of the LPDistLoss with P=2. It is strictly convex.\n\nL(r) = r^2\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    9 │\\                       /│    3 │                   .r/   │\n      │\".                     .\"│      │                 .r\'     │\n      │ \".                   .\" │      │              _./\'       │\n      │  \".                 .\"  │      │_           .r/         _│\n    L │   \".               .\"   │   L\' │         _:/\'            │\n      │    \'\\.           ./\'    │      │       .r\'               │\n      │      \\.         ./      │      │     .r\'                 │\n    0 │        \"-.___.-\"        │   -3 │  _/r\'                   │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -3                        3      -2                        2\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#L2DistLoss-1",
    "page": "Distance-based Losses",
    "title": "L2DistLoss",
    "category": "section",
    "text": "L2DistLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = mid r mid ^2 L(r) = 2 r"
},

{
    "location": "losses/distance/#LossFunctions.LogitDistLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.LogitDistLoss",
    "category": "type",
    "text": "LogitDistLoss <: DistanceLoss\n\nThe distance-based logistic loss for regression. It is strictly convex and Lipschitz continuous.\n\nL(r) = - ln frac4 e^r(1 + e^r)^2\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │                         │    1 │                   _--\'\'\'│\n      │\\                       /│      │                ./\'      │\n      │ \\.                   ./ │      │              ./         │\n      │  \'.                 .\'  │      │_           ./          _│\n    L │   \'.               .\'   │   L\' │           ./            │\n      │     \\.           ./     │      │         ./              │\n      │      \'.         .\'      │      │       ./                │\n    0 │        \'-.___.-\'        │   -1 │___.-\'\'                  │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -3                        3      -4                        4\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#LogitDistLoss-1",
    "page": "Distance-based Losses",
    "title": "LogitDistLoss",
    "category": "section",
    "text": "LogitDistLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = - ln frac4 e^r(1 + e^r)^2 L(r) = tanh left( fracr2 right)"
},

{
    "location": "losses/distance/#LossFunctions.HuberLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.HuberLoss",
    "category": "type",
    "text": "HuberLoss <: DistanceLoss\n\nLoss function commonly used for robustness to outliers. For large values of d it becomes close to the L1DistLoss, while for small values of d it resembles the L2DistLoss. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(r) = begincases fracr^22  quad textif   r  le alpha  alpha  r  - fracalpha^32  quad textotherwise endcases\n\n\n\n              Lossfunction (d=1)               Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │                         │    1 │                .+-------│\n      │                         │      │              ./\'        │\n      │\\.                     ./│      │             ./          │\n      │ \'.                   .\' │      │_           ./          _│\n    L │   \\.               ./   │   L\' │           /\'            │\n      │     \\.           ./     │      │          /\'             │\n      │      \'.         .\'      │      │        ./\'              │\n    0 │        \'-.___.-\'        │   -1 │-------+\'                │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#HuberLoss-1",
    "page": "Distance-based Losses",
    "title": "HuberLoss",
    "category": "section",
    "text": "HuberLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = begincases fracr^22  quad textif  mid r mid le alpha  alpha mid r mid - fracalpha^22  quad textotherwise endcases L(r) = begincases r  quad textif  mid r mid le alpha  alpha cdot textrmsign(r)  quad textotherwise endcases"
},

{
    "location": "losses/distance/#LossFunctions.L1EpsilonInsLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.L1EpsilonInsLoss",
    "category": "type",
    "text": "L1EpsilonInsLoss <: DistanceLoss\n\nThe ϵ-insensitive loss. Typically used in linear support vector regression. It ignores deviances smaller than ϵ, but penalizes larger deviances linarily. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(r) = max  0  r  - epsilon \n\n\n\n              Lossfunction (ϵ=1)               Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │\\                       /│    1 │                  ┌------│\n      │ \\                     / │      │                  |      │\n      │  \\                   /  │      │                  |      │\n      │   \\                 /   │      │_      ___________!     _│\n    L │    \\               /    │   L\' │      |                  │\n      │     \\             /     │      │      |                  │\n      │      \\           /      │      │      |                  │\n    0 │       \\_________/       │   -1 │------┘                  │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -3                        3      -2                        2\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#L1EpsilonInsLoss-1",
    "page": "Distance-based Losses",
    "title": "L1EpsilonInsLoss",
    "category": "section",
    "text": "L1EpsilonInsLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = max  0 mid r mid - epsilon  L(r) = begincases fracr mid r mid   quad textif  epsilon le mid r mid  0  quad textotherwise endcases"
},

{
    "location": "losses/distance/#LossFunctions.L2EpsilonInsLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.L2EpsilonInsLoss",
    "category": "type",
    "text": "L2EpsilonInsLoss <: DistanceLoss\n\nThe quadratic ϵ-insensitive loss. Typically used in linear support vector regression. It ignores deviances smaller than ϵ, but penalizes larger deviances quadratically. It is convex, but not strictly convex.\n\nL(r) = max  0  r  - epsilon ^2\n\n\n\n              Lossfunction (ϵ=0.5)             Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    8 │                         │    1 │                  /      │\n      │:                       :│      │                 /       │\n      │\'.                     .\'│      │                /        │\n      │ \\.                   ./ │      │_         _____/        _│\n    L │  \\.                 ./  │   L\' │         /               │\n      │   \\.               ./   │      │        /                │\n      │    \'\\.           ./\'    │      │       /                 │\n    0 │      \'-._______.-\'      │   -1 │      /                  │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -3                        3      -2                        2\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#L2EpsilonInsLoss-1",
    "page": "Distance-based Losses",
    "title": "L2EpsilonInsLoss",
    "category": "section",
    "text": "L2EpsilonInsLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = max  0 mid r mid - epsilon ^2 L(r) = begincases 2 cdot textrmsign(r) cdot left( mid r mid - epsilon right)  quad textif  epsilon le mid r mid  0  quad textotherwise endcases"
},

{
    "location": "losses/distance/#LossFunctions.PeriodicLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.PeriodicLoss",
    "category": "type",
    "text": "PeriodicLoss <: DistanceLoss\n\nMeasures distance on a circle of specified circumference c.\n\nL(r) = 1 - cos left( frac2 r pic right)\n\n\n\n"
},

{
    "location": "losses/distance/#PeriodicLoss-1",
    "page": "Distance-based Losses",
    "title": "PeriodicLoss",
    "category": "section",
    "text": "PeriodicLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = 1 - cos left ( frac2 r pic right ) L(r) = frac2 pic cdot sin left( frac2r pic right)"
},

{
    "location": "losses/distance/#LossFunctions.QuantileLoss",
    "page": "Distance-based Losses",
    "title": "LossFunctions.QuantileLoss",
    "category": "type",
    "text": "QuantileLoss <: DistanceLoss\n\nThe distance-based quantile loss, also known as pinball loss, can be used to estimate conditional τ-quantiles. It is Lipschitz continuous and convex, but not strictly convex. Furthermore it is symmetric if and only if τ = 1/2.\n\nL(r) = begincases -left( 1 - tau  right) r  quad textif  r  0  tau r  quad textif  r ge 0  endcases\n\n\n\n              Lossfunction (τ=0.7)             Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │\'\\                       │  0.3 │            ┌------------│\n      │  \\.                     │      │            |            │\n      │   \'\\                    │      │_           |           _│\n      │     \\.                  │      │            |            │\n    L │      \'\\              ._-│   L\' │            |            │\n      │        \\.         ..-\'  │      │            |            │\n      │         \'.     _r/\'     │      │            |            │\n    0 │           \'_./\'         │ -0.7 │------------┘            │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -3                        3      -3                        3\n                 ŷ - y                            ŷ - y\n\n\n\n"
},

{
    "location": "losses/distance/#QuantileLoss-1",
    "page": "Distance-based Losses",
    "title": "QuantileLoss",
    "category": "section",
    "text": "QuantileLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(r) = begincases left( 1 - tau right) r  quad textif  r ge 0  - tau r  quad textotherwise  endcases L(r) = begincases 1 - tau  quad textif  r ge 0  - tau  quad textotherwise  endcasesnote: Note\nYou may note that our definition of the QuantileLoss looks different to what one usually sees in other literature. The reason is that we have to correct for the fact that in our case r = haty - y instead of r_textrmusual = y - haty, which means that our definition relates to that in the manner of r = -1 * r_textrmusual.</div>"
},

{
    "location": "losses/margin/#",
    "page": "Margin-based Losses",
    "title": "Margin-based Losses",
    "category": "page",
    "text": "DocTestSetup = quote\n    using LossFunctions\nend<div class=\"loss-docs\">"
},

{
    "location": "losses/margin/#Margin-based-Losses-1",
    "page": "Margin-based Losses",
    "title": "Margin-based Losses",
    "category": "section",
    "text": "Margin-based loss functions are particularly useful for binary classification. In contrast to the distance-based losses, these do not care about the difference between true target and prediction. Instead they penalize predictions based on how well they agree with the sign of the target.This section lists all the subtypes of MarginLoss that are implemented in this package."
},

{
    "location": "losses/margin/#LossFunctions.ZeroOneLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.ZeroOneLoss",
    "category": "type",
    "text": "ZeroOneLoss <: MarginLoss\n\nThe classical classification loss. It penalizes every misclassified observation with a loss of 1 while every correctly classified observation has a loss of 0. It is not convex nor continuous and thus seldom used directly. Instead one usually works with some classification-calibrated surrogate loss, such as L1HingeLoss.\n\nL(a) = begincases 1  quad textif  a  0  0  quad textif  a = 0 endcases\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    1 │------------┐            │    1 │                         │\n      │            |            │      │                         │\n      │            |            │      │                         │\n      │            |            │      │_________________________│\n      │            |            │      │                         │\n      │            |            │      │                         │\n      │            |            │      │                         │\n    0 │            └------------│   -1 │                         │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                y * h(x)                         y * h(x)\n\n\n\n"
},

{
    "location": "losses/margin/#ZeroOneLoss-1",
    "page": "Margin-based Losses",
    "title": "ZeroOneLoss",
    "category": "section",
    "text": "ZeroOneLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = begincases 1  quad textif  a  0  0  quad textotherwise endcases L(a) = 0"
},

{
    "location": "losses/margin/#LossFunctions.PerceptronLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.PerceptronLoss",
    "category": "type",
    "text": "PerceptronLoss <: MarginLoss\n\nThe perceptron loss linearly penalizes every prediction where the resulting agreement <= 0. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(a) = max  0 -a \n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │\\.                       │    0 │            ┌------------│\n      │ \'..                     │      │            |            │\n      │   \\.                    │      │            |            │\n      │     \'.                  │      │            |            │\n    L │      \'.                 │   L\' │            |            │\n      │        \\.               │      │            |            │\n      │         \'.              │      │            |            │\n    0 │           \\.____________│   -1 │------------┘            │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#PerceptronLoss-1",
    "page": "Margin-based Losses",
    "title": "PerceptronLoss",
    "category": "section",
    "text": "PerceptronLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = max  0 - a  L(a) = begincases -1  quad textif  a  0  0  quad textotherwise endcases"
},

{
    "location": "losses/margin/#LossFunctions.L1HingeLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.L1HingeLoss",
    "category": "type",
    "text": "L1HingeLoss <: MarginLoss\n\nThe hinge loss linearly penalizes every predicition where the resulting agreement < 1 . It is Lipschitz continuous and convex, but not strictly convex.\n\nL(a) = max  0 1 - a \n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    3 │\'\\.                      │    0 │                  ┌------│\n      │  \'\'_                    │      │                  |      │\n      │     \\.                  │      │                  |      │\n      │       \'.                │      │                  |      │\n    L │         \'\'_             │   L\' │                  |      │\n      │            \\.           │      │                  |      │\n      │              \'.         │      │                  |      │\n    0 │                \'\'_______│   -1 │------------------┘      │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#L1HingeLoss-1",
    "page": "Margin-based Losses",
    "title": "L1HingeLoss",
    "category": "section",
    "text": "L1HingeLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = max  0 1 - a  L(a) = begincases -1  quad textif  a  1  0  quad textotherwise endcases"
},

{
    "location": "losses/margin/#LossFunctions.SmoothedL1HingeLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.SmoothedL1HingeLoss",
    "category": "type",
    "text": "SmoothedL1HingeLoss <: MarginLoss\n\nAs the name suggests a smoothed version of the L1 hinge loss. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(a) = begincases frac05gamma cdot max  0 1 - a  ^2  quad textif  a ge 1 - gamma  1 - fracgamma2 - a  quad textotherwise endcases\n\n\n\n              Lossfunction (γ=2)               Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │\\.                       │    0 │                 ,r------│\n      │ \'.                      │      │               ./\'       │\n      │   \\.                    │      │              ,/         │\n      │     \'.                  │      │            ./\'          │\n    L │      \'.                 │   L\' │           ,\'            │\n      │        \\.               │      │         ,/              │\n      │          \',             │      │       ./\'               │\n    0 │            \'*-._________│   -1 │______./                 │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#SmoothedL1HingeLoss-1",
    "page": "Margin-based Losses",
    "title": "SmoothedL1HingeLoss",
    "category": "section",
    "text": "SmoothedL1HingeLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = begincases frac12 gamma cdot max  0 1 - a  ^2  quad textif  a ge 1 - gamma  1 - fracgamma2 - a  quad textotherwise endcases L(a) = begincases - frac1gamma cdot max  0 1 - a   quad textif  a ge 1 - gamma  - 1  quad textotherwise endcases"
},

{
    "location": "losses/margin/#LossFunctions.ModifiedHuberLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.ModifiedHuberLoss",
    "category": "type",
    "text": "ModifiedHuberLoss <: MarginLoss\n\nA special (4 times scaled) case of the SmoothedL1HingeLoss with γ=2. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(a) = begincases max  0 1 - a  ^2  quad textif  a ge -1  - 4 a  quad textotherwise endcases\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    5 │    \'.                   │    0 │                .+-------│\n      │     \'.                  │      │              ./\'        │\n      │      \'\\                 │      │             ,/          │\n      │        \\                │      │           ,/            │\n    L │         \'.              │   L\' │         ./              │\n      │          \'.             │      │       ./\'               │\n      │            \\.           │      │______/\'                 │\n    0 │              \'-.________│   -5 │                         │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#ModifiedHuberLoss-1",
    "page": "Margin-based Losses",
    "title": "ModifiedHuberLoss",
    "category": "section",
    "text": "ModifiedHuberLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = begincases max  0 1 - a  ^2  quad textif  a ge -1  - 4 a  quad textotherwise endcases L(a) = begincases - 2 cdot max  0 1 - a   quad textif  a ge -1  - 4  quad textotherwise endcases"
},

{
    "location": "losses/margin/#LossFunctions.DWDMarginLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.DWDMarginLoss",
    "category": "type",
    "text": "DWDMarginLoss <: MarginLoss\n\nThe distance weighted discrimination margin loss. It is a differentiable generalization of the L1HingeLoss that is different than the SmoothedL1HingeLoss. It is Lipschitz continuous and convex, but not strictly convex.\n\nL(a) = begincases 1 - a  quad textif  a ge fracqq+1  frac1a^q fracq^q(q+1)^q+1  quad textotherwise endcases\n\n\n\n              Lossfunction (q=1)               Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │      \".                 │    0 │                     ._r-│\n      │        \\.               │      │                   ./    │\n      │         \',              │      │                 ./      │\n      │           \\.            │      │                 /       │\n    L │            \"\\.          │   L\' │                .        │\n      │              \\.         │      │                /        │\n      │               \":__      │      │               ;         │\n    0 │                   \'\"\"---│   -1 │---------------┘         │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#DWDMarginLoss-1",
    "page": "Margin-based Losses",
    "title": "DWDMarginLoss",
    "category": "section",
    "text": "DWDMarginLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = begincases 1 - a  quad textif  a le fracqq+1  frac1a^q fracq^q(q+1)^q+1  quad textotherwise endcases L(a) = begincases - 1  quad textif  a le fracqq+1  - frac1a^q+1 left( fracqq+1 right)^q+1  quad textotherwise endcases"
},

{
    "location": "losses/margin/#LossFunctions.L2MarginLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.L2MarginLoss",
    "category": "type",
    "text": "L2MarginLoss <: MarginLoss\n\nThe margin-based least-squares loss for classification, which penalizes every prediction where agreement != 1 quadratically. It is locally Lipschitz continuous and strongly convex.\n\nL(a) = left( 1 - a right)^2\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    5 │     .                   │    2 │                       ,r│\n      │     \'.                  │      │                     ,/  │\n      │      \'\\                 │      │                   ,/    │\n      │        \\                │      ├                 ,/      ┤\n    L │         \'.              │   L\' │               ./        │\n      │          \'.             │      │             ./          │\n      │            \\.          .│      │           ./            │\n    0 │              \'-.____.-\' │   -3 │         ./              │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#L2MarginLoss-1",
    "page": "Margin-based Losses",
    "title": "L2MarginLoss",
    "category": "section",
    "text": "L2MarginLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = left( 1 - a right)^2 L(a) = 2 left( a - 1 right)"
},

{
    "location": "losses/margin/#LossFunctions.L2HingeLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.L2HingeLoss",
    "category": "type",
    "text": "L2HingeLoss <: MarginLoss\n\nThe truncated least squares loss quadratically penalizes every predicition where the resulting agreement < 1. It is locally Lipschitz continuous and convex, but not strictly convex.\n\nL(a) = max  0 1 - a ^2\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    5 │     .                   │    0 │                 ,r------│\n      │     \'.                  │      │               ,/        │\n      │      \'\\                 │      │             ,/          │\n      │        \\                │      │           ,/            │\n    L │         \'.              │   L\' │         ./              │\n      │          \'.             │      │       ./                │\n      │            \\.           │      │     ./                  │\n    0 │              \'-.________│   -5 │   ./                    │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#L2HingeLoss-1",
    "page": "Margin-based Losses",
    "title": "L2HingeLoss",
    "category": "section",
    "text": "L2HingeLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = max  0 1 - a  ^2 L(a) = begincases 2 left( a - 1 right)  quad textif  a  1  0  quad textotherwise endcases"
},

{
    "location": "losses/margin/#LossFunctions.LogitMarginLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.LogitMarginLoss",
    "category": "type",
    "text": "LogitMarginLoss <: MarginLoss\n\nThe margin version of the logistic loss. It is infinitely many times differentiable, strictly convex, and Lipschitz continuous.\n\nL(a) = ln (1 + e^-a)\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │ \\.                      │    0 │                  ._--/\"\"│\n      │   \\.                    │      │               ../\'      │\n      │     \\.                  │      │              ./         │\n      │       \\..               │      │            ./\'          │\n    L │         \'-_             │   L\' │          .,\'            │\n      │            \'-_          │      │         ./              │\n      │               \'\\-._     │      │      .,/\'               │\n    0 │                    \'\"\"*-│   -1 │__.--\'\'                  │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -4                        4\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#LogitMarginLoss-1",
    "page": "Margin-based Losses",
    "title": "LogitMarginLoss",
    "category": "section",
    "text": "LogitMarginLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = ln (1 + e^-a) L(a) = - frac11 + e^a"
},

{
    "location": "losses/margin/#LossFunctions.ExpLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.ExpLoss",
    "category": "type",
    "text": "ExpLoss <: MarginLoss\n\nThe margin-based exponential loss for classification, which penalizes every prediction exponentially. It is infinitely many times differentiable, locally Lipschitz continuous and strictly convex, but not clipable.\n\nL(a) = e^-a\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    5 │  \\.                     │    0 │               _,,---:\'\"\"│\n      │   l                     │      │           _r/\"\'         │\n      │    l.                   │      │        .r/\'             │\n      │     \":                  │      │      .r\'                │\n    L │       \\.                │   L\' │     ./                  │\n      │        \"\\..             │      │    .\'                   │\n      │           \'\":,_         │      │   ,\'                    │\n    0 │                \"\"---:.__│   -5 │  ./                     │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#ExpLoss-1",
    "page": "Margin-based Losses",
    "title": "ExpLoss",
    "category": "section",
    "text": "ExpLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = e^-a L(a) = - e^-a"
},

{
    "location": "losses/margin/#LossFunctions.SigmoidLoss",
    "page": "Margin-based Losses",
    "title": "LossFunctions.SigmoidLoss",
    "category": "type",
    "text": "SigmoidLoss <: MarginLoss\n\nContinuous loss which penalizes every prediction with a loss within in the range (0,2). It is infinitely many times differentiable, Lipschitz continuous but nonconvex.\n\nL(a) = 1 - tanh(a)\n\n\n\n              Lossfunction                     Derivative\n      ┌────────────┬────────────┐      ┌────────────┬────────────┐\n    2 │\"\"\'--,.                  │    0 │..                     ..│\n      │      \'\\.                │      │ \"\\.                 ./\" │\n      │         \'.              │      │    \',             ,\'    │\n      │           \\.            │      │      \\           /      │\n    L │            \"\\.          │   L\' │       \\         /       │\n      │              \\.         │      │        \\.     ./        │\n      │                \\,       │      │         \\.   ./         │\n    0 │                  \'\"-:.__│   -1 │          \',_,\'          │\n      └────────────┴────────────┘      └────────────┴────────────┘\n      -2                        2      -2                        2\n                 y ⋅ ŷ                            y ⋅ ŷ\n\n\n\n"
},

{
    "location": "losses/margin/#SigmoidLoss-1",
    "page": "Margin-based Losses",
    "title": "SigmoidLoss",
    "category": "section",
    "text": "SigmoidLossLossfunction Derivative\n(Image: loss) (Image: deriv)\nL(a) = 1 - tanh(a) L(a) = - textrmsech^2 (a)</div>"
},

{
    "location": "advanced/extend/#",
    "page": "Altering existing Losses",
    "title": "Altering existing Losses",
    "category": "page",
    "text": "DocTestSetup = quote\n    using LossFunctions\nend"
},

{
    "location": "advanced/extend/#Altering-existing-Losses-1",
    "page": "Altering existing Losses",
    "title": "Altering existing Losses",
    "category": "section",
    "text": "There are situations in which one wants to work with slightly altered versions of specific loss functions. This package provides two generic ways to create such meta losses for specific families of loss functions.Scaling a supervised loss by a constant real number. This is done at compile time and can in some situations even lead to simpler code (e.g. in the case of the derivative for a L2DistLoss)\nWeighting the classes of a margin-based loss differently in order to better deal with unbalanced binary classification problems."
},

{
    "location": "advanced/extend/#LearnBase.scaled",
    "page": "Altering existing Losses",
    "title": "LearnBase.scaled",
    "category": "function",
    "text": "scaled(loss::SupervisedLoss, K)\n\nReturns a version of loss that is uniformly scaled by K. This function dispatches on the type of loss in order to choose the appropriate type of scaled loss that will be used as the decorator. For example, if typeof(loss) <: DistanceLoss then the given loss will be boxed into a ScaledDistanceLoss.\n\nNote: If typeof(K) <: Number, then this method will poison the type-inference of the calling scope. This is because K will be promoted to a type parameter. For a typestable version use the following signature: scaled(loss, Val(K))\n\n\n\n\n\n"
},

{
    "location": "advanced/extend/#Scaling-a-Supervised-Loss-1",
    "page": "Altering existing Losses",
    "title": "Scaling a Supervised Loss",
    "category": "section",
    "text": "It is quite common in machine learning courses to define the least squares loss as frac12 (haty - y)^2, while this package implements that type of loss as an L_2 distance loss using (haty - y)^2, i.e. without the constant scale factor.For situations in which one wants a scaled version of an existing loss type, we provide the concept of a scaled loss. The difference is literally only a constant real number that gets multiplied to the existing implementation of the loss function (and derivatives).scaledjulia> lsloss = 1/2 * L2DistLoss()\nScaledDistanceLoss{LPDistLoss{2},0.5}(LPDistLoss{2}())\n\njulia> value(L2DistLoss(), 0.0, 4.0)\n16.0\n\njulia> value(lsloss, 0.0, 4.0)\n8.0While the resulting loss is of the same basic family as the original loss (i.e. margin-based or distance-based), it is not a sub-type of it.julia> typeof(lsloss) <: DistanceLoss\ntrue\n\njulia> typeof(lsloss) <: L2DistLoss\nfalseAs you have probably noticed, the constant scale factor gets promoted to a type-parameter. This can be quite an overhead when done on the fly every time the loss value is computed. To avoid this one can make use of Val to specify the scale factor in a type-stable manner.julia> lsloss = scaled(L2DistLoss(), Val(0.5))\nScaledDistanceLoss{LPDistLoss{2},0.5}(LPDistLoss{2}())Storing the scale factor as a type-parameter instead of a member variable has some nice advantages. For one it makes it possible to define new types of losses using simple type-aliases.julia> const LeastSquaresLoss = LossFunctions.ScaledDistanceLoss{L2DistLoss,0.5}\nScaledDistanceLoss{LPDistLoss{2},0.5}\n\njulia> value(LeastSquaresLoss(), 0.0, 4.0)\n8.0Furthermore, it allows the compiler to do some quite convenient optimizations if possible. For example the compiler is able to figure out that the derivative simplifies for our newly defined LeastSquaresLoss, because 1/2 * 2 cancels each other. This is accomplished using the power of @fastmath.julia> @code_llvm deriv(L2DistLoss(), 0.0, 4.0)\ndefine double @julia_deriv_71652(double, double) #0 {\ntop:\n  %2 = fsub double %1, %0\n  %3 = fmul double %2, 2.000000e+00\n  ret double %3\n}\n\njulia> @code_llvm deriv(LeastSquaresLoss(), 0.0, 4.0)\ndefine double @julia_deriv_71659(double, double) #0 {\ntop:\n  %2 = fsub double %1, %0\n  ret double %2\n}"
},

{
    "location": "advanced/extend/#LossFunctions.weightedloss",
    "page": "Altering existing Losses",
    "title": "LossFunctions.weightedloss",
    "category": "function",
    "text": "weightedloss(loss, weight)\n\nReturns a weighted version of loss for which the value of the positive class is changed to be weight times its original, and the negative class 1 - weight times its original respectively.\n\nNote: If typeof(weight) <: Number, then this method will poison the type-inference of the calling scope. This is because weight will be promoted to a type parameter. For a typestable version use the following signature: weightedloss(loss, Val(weight))\n\n\n\n\n\n"
},

{
    "location": "advanced/extend/#Reweighting-a-Margin-Loss-1",
    "page": "Altering existing Losses",
    "title": "Reweighting a Margin Loss",
    "category": "section",
    "text": "It is not uncommon in classification scenarios to find yourself working with in-balanced data sets, where one class has much more observations than the other one. There are different strategies to deal with this kind of problem. The approach that this package provides is to weight the loss for the classes differently. This basically means that we penalize mistakes in one class more than mistakes in the other class. More specifically we scale the loss of the positive class by the weight-factor w and the loss of the negative class with 1-w.if target > 0\n    w * loss(target, output)\nelse\n    (1-w) * loss(target, output)\nendInstead of providing special functions to compute a class-weighted loss, we instead expose a generic way to create new weighted versions of already existing unweighted losses. This way, every existing subtype of MarginLoss can be re-weighted arbitrarily. Furthermore, it allows every algorithm that expects a binary loss to work with weighted binary losses as well.weightedlossjulia> myloss = weightedloss(HingeLoss(), 0.8)\nWeightedBinaryLoss{L1HingeLoss,0.8}(L1HingeLoss())\n\njulia> value(myloss, 1.0, -4.0) # positive class\n4.0\n\njulia> value(HingeLoss(), 1.0, -4.0)\n5.0\n\njulia> value(myloss, -1.0, 4.0) # negative class\n0.9999999999999998\n\njulia> value(HingeLoss(), -1.0, 4.0)\n5.0Note that the scaled version of a margin-based loss does not anymore belong to the family of margin-based losses itself. In other words the resulting loss is neither a subtype of MarginLoss, nor of the original type of loss.julia> typeof(myloss) <: MarginLoss\nfalse\n\njulia> typeof(myloss) <: HingeLoss\nfalseSimilar to scaled losses, the constant weight factor gets promoted to a type-parameter. This can be quite an overhead when done on the fly every time the loss value is computed. To avoid this one can make use of Val to specify the scale factor in a type-stable manner.julia> myloss = weightedloss(HingeLoss(), Val(0.8))\nWeightedBinaryLoss{L1HingeLoss,0.8}(L1HingeLoss())Storing the scale factor as a type-parameter instead of a member variable has a nice advantage. It makes it possible to define new types of losses using simple type-aliases.julia> const MyWeightedHingeLoss = LossFunctions.WeightedBinaryLoss{HingeLoss,0.8}\nWeightedBinaryLoss{L1HingeLoss,0.8}\n\njulia> value(MyWeightedHingeLoss(), 1.0, -4.0)\n4.0"
},

{
    "location": "advanced/developer/#",
    "page": "Developer Documentation",
    "title": "Developer Documentation",
    "category": "page",
    "text": ""
},

{
    "location": "advanced/developer/#Developer-Documentation-1",
    "page": "Developer Documentation",
    "title": "Developer Documentation",
    "category": "section",
    "text": "In this part of the documentation we will discuss some of the internal design aspects of this library. Consequently, the target audience of this section and its sub-sections is primarily people interested in contributing to this package. As such, the information provided here should be of little to no relevance for users interested in simply applying the package."
},

{
    "location": "advanced/developer/#LearnBase.SupervisedLoss",
    "page": "Developer Documentation",
    "title": "LearnBase.SupervisedLoss",
    "category": "type",
    "text": "A loss is considered supervised, if all the information needed to compute L(features, targets, outputs) are contained in targets and outputs, and thus allows for the simplification L(targets, outputs).\n\n\n\n\n\n"
},

{
    "location": "advanced/developer/#LearnBase.DistanceLoss",
    "page": "Developer Documentation",
    "title": "LearnBase.DistanceLoss",
    "category": "type",
    "text": "A supervised loss that can be simplified to L(targets, outputs) = L(targets - outputs) is considered distance-based.\n\n\n\n\n\n"
},

{
    "location": "advanced/developer/#LearnBase.MarginLoss",
    "page": "Developer Documentation",
    "title": "LearnBase.MarginLoss",
    "category": "type",
    "text": "A supervised loss, where the targets are in {-1, 1}, and which can be simplified to L(targets, outputs) = L(targets * outputs) is considered margin-based.\n\n\n\n\n\n"
},

{
    "location": "advanced/developer/#Abstract-Types-1",
    "page": "Developer Documentation",
    "title": "Abstract Types",
    "category": "section",
    "text": "We have seen in previous sections, that many families of loss functions are implemented as immutable types with free parameters. An example for such a family is the L1EpsilonInsLoss, which represents all the epsilon-insensitive loss-functions for each possible value of epsilon.Aside from these special families, there a handful of more generic families that between them contain almost all of the loss functions this package implements. These families are defined as abstract types in the type tree. Their main purpose is two-fold:From an end-user\'s perspective, they are most useful for dispatching on the particular kind of prediction problem that they are intended for (regression vs classification).\nForm an implementation perspective, these abstract types allow us to implement shared functionality and fall-back methods, or even allow for a simpler implementation.Most of the implemented loss functions fall under the umbrella of supervised losses. As such, we barely mention other types of losses anywhere in this documentation.SupervisedLossThere are two interesting sub-families of supervised loss functions.  One of these families is called distance-based. All losses that belong to this family are implemented as subtype of the abstract type DistanceLoss, which itself is subtype of SupervisedLoss.DistanceLossThe second core sub-family of supervised losses is called margin-based. All loss functions that belong to this family are implemented as subtype of the abstract type MarginLoss, which itself is subtype of SupervisedLoss.MarginLoss"
},

{
    "location": "advanced/developer/#Shared-Interface-1",
    "page": "Developer Documentation",
    "title": "Shared Interface",
    "category": "section",
    "text": "Each of the three abstract types listed above serves a purpose other than dispatch. All losses that belong to the same family share functionality to some degree. For example all subtypes of SupervisedLoss share the same implementations for the vectorized versions of value and deriv.More interestingly, the abstract types DistanceLoss and MarginLoss, serve an additional purpose aside from shared functionality. We have seen in the background section what it is that makes a loss margin-based or distance-based. Without repeating the definition let us state that it boils down to the existence of a representing function psi, which allows to compute a loss using a unary function instead of a binary one. Indeed, all the subtypes of DistanceLoss and MarginLoss are implemented in the unary form of their representing function."
},

{
    "location": "advanced/developer/#Distance-based-Losses-1",
    "page": "Developer Documentation",
    "title": "Distance-based Losses",
    "category": "section",
    "text": "Supervised losses that can be expressed as a univariate function of output - target are referred to as distance-based losses. Distance-based losses are typically utilized for regression problems. That said, there are also other losses that are useful for regression problems that don\'t fall into this category, such as the PeriodicLoss."
},

{
    "location": "advanced/developer/#Margin-based-Losses-1",
    "page": "Developer Documentation",
    "title": "Margin-based Losses",
    "category": "section",
    "text": "Margin-based losses are supervised losses where the values of the targets are restricted to be in 1-1, and which can be expressed as a univariate function output * target."
},

{
    "location": "indices/#",
    "page": "Indices",
    "title": "Indices",
    "category": "page",
    "text": ""
},

{
    "location": "indices/#Functions-1",
    "page": "Indices",
    "title": "Functions",
    "category": "section",
    "text": "Order   = [:function]"
},

{
    "location": "indices/#Types-1",
    "page": "Indices",
    "title": "Types",
    "category": "section",
    "text": "Order   = [:type]"
},

{
    "location": "acknowledgements/#",
    "page": "Acknowledgements",
    "title": "Acknowledgements",
    "category": "page",
    "text": ""
},

{
    "location": "acknowledgements/#Acknowledgements-1",
    "page": "Acknowledgements",
    "title": "Acknowledgements",
    "category": "section",
    "text": "The basic design of this package is heavily modelled after the loss-related definitions in [STEINWART2008].We would also like to mention that some early inspiration was drawn from EmpiricalRisks.jl"
},

{
    "location": "acknowledgements/#References-1",
    "page": "Acknowledgements",
    "title": "References",
    "category": "section",
    "text": "[STEINWART2008]: Steinwart, Ingo, and Andreas Christmann. \"Support vector machines\". Springer Science & Business Media, 2008."
},

{
    "location": "LICENSE/#",
    "page": "LICENSE",
    "title": "LICENSE",
    "category": "page",
    "text": ""
},

{
    "location": "LICENSE/#LICENSE-1",
    "page": "LICENSE",
    "title": "LICENSE",
    "category": "section",
    "text": "using Markdown\nMarkdown.parse_file(joinpath(@__DIR__, \"..\", \"..\", \"LICENSE.md\"))"
},

]}
