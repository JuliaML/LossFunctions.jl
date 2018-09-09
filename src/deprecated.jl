@deprecate ScaledMarginLoss(loss, ::Type{Val{K}}) where {K}  ScaledMarginLoss(loss, Val(K))
@deprecate ScaledDistanceLoss(loss, ::Type{Val{K}}) where {K}  ScaledDistanceLoss(loss, Val(K))
@deprecate ScaledSupervisedLoss(loss, ::Type{Val{K}}) where {K}  ScaledSupervisedLoss(loss, Val(K))
@deprecate ((*)(::Type{Val{K}}, loss::Loss) where {K}) (*)(Val(K), loss)
@deprecate scaled(loss, ::Type{Val{K}}) where {K}  scaled(loss, Val(K))
@deprecate weightedloss(loss::Loss, ::Type{Val{W}}) where {W}  weightedloss(loss, Val(W))
@deprecate WeightedBinaryLoss(loss, ::Type{Val{W}}) where {W}  WeightedBinaryLoss(loss, Val(W))

Base.@deprecate_binding AverageMode AggregateMode
module AvgMode
    import ..LossFunctions.AggMode
    Base.@deprecate_binding None AggMode.None
    Base.@deprecate_binding Sum AggMode.Sum
    Base.@deprecate_binding Mean AggMode.Mean
    Base.@deprecate_binding Macro AggMode.Mean
    Base.@deprecate_binding Micro AggMode.Mean
    Base.@deprecate_binding WeightedSum AggMode.WeightedSum
    Base.@deprecate_binding WeightedMean AggMode.WeightedMean
end
