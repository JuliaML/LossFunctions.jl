x = randn(10)
X = randn(10, 20)
w = randn(10)
wb = randn(11)
len_wb = length(wb) - 1

# ==========================================================================

msg("Test linear prediction function without bias")

pred = LinearPredictor(0)
@test_approx_eq value(pred, x, w) dot(x, w)
@test_approx_eq grad(pred, x, w) x

val = 0.
for i = 1:length(x)
  val += value(pred, x[i], w[i])
end
@test_approx_eq val dot(x, w)

buffer = zeros(length(x))
for i = 1:length(x)
  buffer[i] = deriv(pred, x[i], w[i])
end
@test_approx_eq buffer x

@test_approx_eq value(pred, X, w) w'X
buffer = zeros(1, size(X, 2))
@test_approx_eq value!(buffer, pred, X, w) w'X
@test_approx_eq buffer w'X

@test_approx_eq grad(pred, X, w) X
buffer = zeros(size(X))
@test_approx_eq grad!(buffer, pred, X, w) X
@test_approx_eq buffer X

# ==========================================================================

msg("Test linear prediction function with bias")

bias = 0.5
pred = LinearPredictor(bias)
@test_approx_eq value(pred, x, wb) (dot(x, wb[1:len_wb]) + bias * wb[end])
@test_approx_eq grad(pred, x, wb) [x; bias]

val = bias * wb[end]
for i = 1:length(x)
  val += value(pred, x[i], wb[i])
end
@test_approx_eq val (dot(x, wb[1:len_wb]) + bias * wb[end])

@test_approx_eq value(pred, X, wb) (wb[1:len_wb]'X + bias * wb[end])
buffer = zeros(1, size(X, 2))
@test_approx_eq value!(buffer, pred, X, wb) (wb[1:len_wb]'X + bias * wb[end])
@test_approx_eq buffer (wb[1:len_wb]'X + bias * wb[end])

@test_approx_eq grad(pred, X, wb) vcat(X, bias .* ones(1, size(X,2)))
buffer = zeros(size(X,1) + 1, size(X,2))
@test_approx_eq grad!(buffer, pred, X, wb) vcat(X, bias .* ones(1, size(X,2)))
@test_approx_eq buffer vcat(X, bias .* ones(1, size(X,2)))
