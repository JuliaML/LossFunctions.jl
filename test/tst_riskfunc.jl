@testset "Test linear regression with value_grad_fun" begin
    w = [1, 10]
    x, y = noisy_poly(w, -10:.2:10, noise = .5)
    X = x'

    # Set hyper parameters
    θ = zeros(length(w))
    α = 0.005
    maxIter = 1000

    loss = LossFunctions.L2DistLoss()
    pred = LinearPredictor(bias = 1)
    risk = EmpiricalRisk(pred, loss)
    riskfunc = RiskFunctional(risk, X, y)

    fg! = value_grad_fun(riskfunc)

    J = 0.
    ▽ = zeros(θ)
    for i = 1:maxIter
        J = fg!(θ, ▽)
        θ = θ - α .* ▽
    end

    @test J < 0.3
    @test sumabs(w - θ) < .15
    mp = scatterplot(x, y, color = :blue, height = 5, margin = 5,
                     title = "Linreg with value_grad_fun")
    lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
    print(mp)
end

@testset "Test linear regression with value_fun and grad_fun" begin
    k = 5
    x, y = noisy_sin(100; noise = .1)
    X = expand_poly(x, degree = 4)
    rescale!(X)

    # Set hyper parameters
    θ = zeros(k)
    α = 0.2
    maxIter = 5000

    loss = LossFunctions.L2DistLoss()
    pred = LinearPredictor(bias = 1)
    risk = EmpiricalRisk(pred, loss)
    riskfunc = RiskFunctional(risk, X, y)

    f = value_fun(riskfunc)
    g! = grad_fun(riskfunc)

    J = 0.
    ▽ = zeros(θ)
    for i = 1:maxIter
        J = f(θ)
        g!(θ, ▽)
        θ = θ - α .* ▽
    end

    @test J < 0.04
    mp = scatterplot(x, y, color = :blue, height = 5,
                     title = "Linreg with value_fun, grad_fun")
    lineplot!(mp, x, vec(value(pred, X, θ)), color = :red)
    print(mp)
end

