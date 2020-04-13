_loss_xguide(loss::MarginLoss) = "y * h(x)"
_loss_xguide(loss::DistanceLoss) = "h(x) - y"
_loss_yguide(loss::SupervisedLoss) = "L("*_loss_xguide(loss)*")"

@recipe function plot(loss::SupervisedLoss, range=-2:0.05:2; fun=value)
    xguide --> _loss_xguide(loss)
    yguide --> _loss_yguide(loss)
    label  --> string(loss)
    l(a) = fun(loss, a)
    l, range
end
