abstract EncodedStatisticalModel{E<:ClassEncoding} <: StatisticalModel
abstract EncodedRegressionModel{E<:ClassEncoding} <: RegressionModel

# ==========================================================================

immutable BoxedEncodedStatisticalModel{TModel, TEnc<:ClassEncoding} <: EncodedStatisticalModel{TEnc}
    model::TModel
    encoding::TEnc
end

function BoxedEncodedStatisticalModel(model::StatisticalModel, encoding::BinaryClassEncoding = SignedClassEncoding([-1,1]))
    BoxedEncodedStatisticalModel{typeof(model),typeof(encoding)}(model, encoding)
end

# ==========================================================================

immutable BoxedEncodedRegressionModel{TModel, TEnc<:ClassEncoding} <: EncodedRegressionModel{TEnc}
    model::TModel
    encoding::TEnc
end

function BoxedEncodedRegressionModel(model::RegressionModel, encoding::BinaryClassEncoding = SignedClassEncoding([-1,1]))
    BoxedEncodedRegressionModel{typeof(model),typeof(encoding)}(model, encoding)
end

# ==========================================================================

typealias BoxedEncodedModel Union{BoxedEncodedStatisticalModel, BoxedEncodedRegressionModel}

# ==========================================================================

for (M, BM) in ((:EncodedStatisticalModel, :BoxedEncodedStatisticalModel),
                (:EncodedRegressionModel, :BoxedEncodedRegressionModel))
    for ENC = (:ZeroOneClassEncoding, :SignedClassEncoding,
               :MultivalueClassEncoding, :OneOfKClassEncoding)
        @eval begin
            function StatsBase.fit{T<:$M{$ENC}}(
                    mt::Type{T},
                    X, y::AbstractVector,
                    args...; nargs...)
                encoding = ($ENC)(y)
                model = fit(mt, X, labelencode(encoding, y), args...; nargs...)
                ($BM)(model, encoding)
            end
        end
    end
end

# ==========================================================================

labels(model::BoxedEncodedModel) = labels(model.encoding)

function StatsBase.predict(model::BoxedEncodedModel, args...)
    y = predict(model.model, args...)
    labeldecode(model.encoding, y)
end

for op = (:coef, :confint, :deviance, :loglikelihood,
          :coeftable, :nobs, :stderr, :vcov)
    @eval ($op)(model::BoxedEncodedModel, args...) = ($op)(model.model, args...)
end

for op = (:residuals, :model_response, :predict, :predict!)
    @eval ($op)(model::BoxedEncodedRegressionModel, args...) = ($op)(model.model, args...)
end

function Base.show(io::IO, model::EncodedStatisticalModel)
    show(io, model.model)
end
