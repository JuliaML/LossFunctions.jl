struct TstVanillaLoss <: SupervisedLoss end

@testset "Fallback implementations; not prior knowledge" begin
  @test isminimizable(TstVanillaLoss()) == false
  @test isdifferentiable(TstVanillaLoss()) == false
  @test isdifferentiable(TstVanillaLoss(), 0) == false
  @test istwicedifferentiable(TstVanillaLoss()) == false
  @test istwicedifferentiable(TstVanillaLoss(), 0) == false

  @test isstronglyconvex(TstVanillaLoss()) == false
  @test isstrictlyconvex(TstVanillaLoss()) == false
  @test isconvex(TstVanillaLoss()) == false

  @test isnemitski(TstVanillaLoss()) == false
  @test islipschitzcont(TstVanillaLoss()) == false
  @test islocallylipschitzcont(TstVanillaLoss()) == false
  @test isclipable(TstVanillaLoss()) == false
  @test ismarginbased(TstVanillaLoss()) == false
  @test isdistancebased(TstVanillaLoss()) == false
  @test issymmetric(TstVanillaLoss()) == false
end

struct TstStronglyConvexLoss <: SupervisedLoss end
LossFunctions.isstronglyconvex(::TstStronglyConvexLoss) = true

@testset "Fallback implementations; strongly convex" begin
  @test isminimizable(TstStronglyConvexLoss()) == true

  @test isdifferentiable(TstStronglyConvexLoss()) == false
  @test isdifferentiable(TstStronglyConvexLoss(), 0) == false
  @test istwicedifferentiable(TstStronglyConvexLoss()) == false
  @test istwicedifferentiable(TstStronglyConvexLoss(), 0) == false

  @test isstronglyconvex(TstStronglyConvexLoss()) == true
  @test isstrictlyconvex(TstStronglyConvexLoss()) == true
  @test isconvex(TstStronglyConvexLoss()) == true

  @test isnemitski(TstStronglyConvexLoss()) == true
  @test islipschitzcont(TstStronglyConvexLoss()) == false
  @test islocallylipschitzcont(TstStronglyConvexLoss()) == true
  @test isclipable(TstStronglyConvexLoss()) == false
  @test ismarginbased(TstStronglyConvexLoss()) == false
  @test isdistancebased(TstStronglyConvexLoss()) == false
  @test issymmetric(TstStronglyConvexLoss()) == false
end

struct TstTwiceDiffLoss <: SupervisedLoss end
LossFunctions.istwicedifferentiable(::TstTwiceDiffLoss) = true

@testset "Fallback implementations; twice differentiable" begin
  @test isminimizable(TstTwiceDiffLoss()) == false

  @test isdifferentiable(TstTwiceDiffLoss()) == true
  @test isdifferentiable(TstTwiceDiffLoss(), 0) == true
  @test istwicedifferentiable(TstTwiceDiffLoss()) == true
  @test istwicedifferentiable(TstTwiceDiffLoss(), 0) == true

  @test isstronglyconvex(TstTwiceDiffLoss()) == false
  @test isstrictlyconvex(TstTwiceDiffLoss()) == false
  @test isconvex(TstTwiceDiffLoss()) == false

  @test isnemitski(TstTwiceDiffLoss()) == false
  @test islipschitzcont(TstTwiceDiffLoss()) == false
  @test islocallylipschitzcont(TstTwiceDiffLoss()) == false
  @test isclipable(TstTwiceDiffLoss()) == false
  @test ismarginbased(TstTwiceDiffLoss()) == false
  @test isdistancebased(TstTwiceDiffLoss()) == false
  @test issymmetric(TstTwiceDiffLoss()) == false
end

struct TstMarginLoss <: MarginLoss end

@testset "Fallback implementations; margin-based" begin
  @test isminimizable(TstMarginLoss()) == false

  @test isdifferentiable(TstMarginLoss()) == false
  @test isdifferentiable(TstMarginLoss(), 0) == false
  @test istwicedifferentiable(TstMarginLoss()) == false
  @test istwicedifferentiable(TstMarginLoss(), 0) == false

  @test isstronglyconvex(TstMarginLoss()) == false
  @test isstrictlyconvex(TstMarginLoss()) == false
  @test isconvex(TstMarginLoss()) == false

  @test isnemitski(TstMarginLoss()) == true
  @test islipschitzcont(TstMarginLoss()) == false
  @test islocallylipschitzcont(TstMarginLoss()) == false
  @test isclipable(TstMarginLoss()) == false
  @test ismarginbased(TstMarginLoss()) == true
  @test isdistancebased(TstMarginLoss()) == false
  @test issymmetric(TstMarginLoss()) == false

  @test isclasscalibrated(TstMarginLoss()) == false
  @test isfishercons(TstMarginLoss()) == false
  @test isunivfishercons(TstMarginLoss()) == false
end

struct TstDistanceLoss <: DistanceLoss end

@testset "Fallback implementations; distance-based" begin
  @test isminimizable(TstDistanceLoss()) == false

  @test isdifferentiable(TstDistanceLoss()) == false
  @test isdifferentiable(TstDistanceLoss(), 0) == false
  @test istwicedifferentiable(TstDistanceLoss()) == false
  @test istwicedifferentiable(TstDistanceLoss(), 0) == false

  @test isstronglyconvex(TstDistanceLoss()) == false
  @test isstrictlyconvex(TstDistanceLoss()) == false
  @test isconvex(TstDistanceLoss()) == false

  @test isnemitski(TstDistanceLoss()) == false
  @test islipschitzcont(TstDistanceLoss()) == false
  @test islocallylipschitzcont(TstDistanceLoss()) == false
  @test isclipable(TstDistanceLoss()) == true
  @test ismarginbased(TstDistanceLoss()) == false
  @test isdistancebased(TstDistanceLoss()) == true
  @test issymmetric(TstDistanceLoss()) == false
end

# --------------------------------------------------------------

@testset "LPDistLoss{0.5}" begin
  loss = LPDistLoss(0.5)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == false
  @test isdifferentiable(loss, 0) == false
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == false

  @test isnemitski(loss) == false
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == false
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "L1DistLoss" begin
  loss = L1DistLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == false
  @test isdifferentiable(loss, 0) == false
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "L2DistLoss" begin
  loss = L2DistLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true

  @test isstronglyconvex(loss) == true
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "LPDistLoss{3}" begin
  loss = LPDistLoss(3)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true

  @test isstronglyconvex(loss) == true
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "HuberLoss(1)" begin
  loss = HuberLoss(1)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == false

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "L1EpsilonInsLoss(1)" begin
  loss = L1EpsilonInsLoss(1)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == false
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == false
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 0) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "L2EpsilonInsLoss(1)" begin
  loss = L2EpsilonInsLoss(1)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == false

  @test isstronglyconvex(loss) == true
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "LogitDistLoss()" begin
  loss = LogitDistLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

@testset "QuantileLoss" begin
  l1 = QuantileLoss(0.5)
  l2 = QuantileLoss(0.7)

  @test issymmetric(l1) == true
  @test issymmetric(l2) == false

  @test isminimizable(l2) == true

  @test isdifferentiable(l2) == false
  @test isdifferentiable(l2, 0) == false
  @test isdifferentiable(l2, 1) == true
  @test istwicedifferentiable(l2) == false
  @test istwicedifferentiable(l2, 0) == false
  @test istwicedifferentiable(l2, 1) == true

  @test isstronglyconvex(l2) == false
  @test isstrictlyconvex(l2) == false
  @test isconvex(l2) == true

  # @test isnemitski(l2) == ?
  @test islipschitzcont(l2) == true
  @test islocallylipschitzcont(l2) == true
  # @test isclipable(l2) == ?
  @test ismarginbased(l2) == false
  @test isdistancebased(l2) == true
end

@testset "LogCoshLoss" begin
  loss = LogCoshLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true

  @test isstronglyconvex(loss) == true
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  #@test isnemitski(loss)              == ?
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == false
  @test isdistancebased(loss) == true
  @test issymmetric(loss) == true
end

# --------------------------------------------------------------

@testset "ZeroOneLoss" begin
  loss = ZeroOneLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == false
  @test isdifferentiable(loss, 0) == false
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == false

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
end

@testset "PerceptronLoss" begin
  loss = PerceptronLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == false
  @test isdifferentiable(loss, 0) == false
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 0) == false
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == false
end

@testset "LogitMarginLoss" begin
  loss = LogitMarginLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == false
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == true
end

@testset "L1HingeLoss" begin
  loss = L1HingeLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == false
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == false
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == false

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == false
end

@testset "L2HingeLoss" begin
  loss = L2HingeLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == false

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == true
end

@testset "SmoothedL1HingeLoss" begin
  loss = SmoothedL1HingeLoss(2)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, -1) == false
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == false
  @test istwicedifferentiable(loss, 2) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
end

@testset "ModifiedHuberLoss" begin
  loss = ModifiedHuberLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == false
  @test istwicedifferentiable(loss, -1) == false
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == false
  @test istwicedifferentiable(loss, 2) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
end

@testset "L2MarginLoss" begin
  loss = L2MarginLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == true
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == true
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == true
end

@testset "ExpLoss" begin
  loss = ExpLoss()

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == true
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == false
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == false
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == true
end

@testset "SigmoidLoss" begin
  loss = SigmoidLoss()

  @test isminimizable(loss) == false

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == false

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == false
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == true
end

@testset "DWDMarginLoss" begin
  loss = DWDMarginLoss(2)

  @test isminimizable(loss) == true

  @test isdifferentiable(loss) == true
  @test isdifferentiable(loss, 0) == true
  @test isdifferentiable(loss, 1) == true
  @test istwicedifferentiable(loss) == true
  @test istwicedifferentiable(loss, 0) == true
  @test istwicedifferentiable(loss, 1) == true

  @test isstronglyconvex(loss) == false
  @test isstrictlyconvex(loss) == false
  @test isconvex(loss) == true

  @test isnemitski(loss) == true
  @test islipschitzcont(loss) == true
  @test islocallylipschitzcont(loss) == true
  @test isclipable(loss) == false
  @test ismarginbased(loss) == true
  @test isdistancebased(loss) == false
  @test issymmetric(loss) == false
  @test isclasscalibrated(loss) == true
  @test isfishercons(loss) == true
  @test isunivfishercons(loss) == true
end

# --------------------------------------------------------------

function compare_losses(l1, l2, ccal=true)
  @test isminimizable(l1) == isminimizable(l2)

  @test isdifferentiable(l1) == isdifferentiable(l2)
  @test isdifferentiable(l1, 0) == isdifferentiable(l2, 0)
  @test isdifferentiable(l1, 1) == isdifferentiable(l2, 1)
  @test istwicedifferentiable(l1) == istwicedifferentiable(l2)
  @test istwicedifferentiable(l1, 0) == istwicedifferentiable(l2, 0)
  @test istwicedifferentiable(l1, 1) == istwicedifferentiable(l2, 1)
  @test istwicedifferentiable(l1, 2) == istwicedifferentiable(l2, 2)

  @test isstronglyconvex(l1) == isstronglyconvex(l2)
  @test isstrictlyconvex(l1) == isstrictlyconvex(l2)
  @test isconvex(l1) == isconvex(l2)

  @test isnemitski(l1) == isnemitski(l2)
  @test islipschitzcont(l1) == islipschitzcont(l2)
  @test islocallylipschitzcont(l1) == islocallylipschitzcont(l2)
  @test isclipable(l1) == isclipable(l2)
  @test ismarginbased(l1) == ismarginbased(l2)
  @test isdistancebased(l1) == isdistancebased(l2)
  @test issymmetric(l1) == issymmetric(l2)
  @test (ccal && isclasscalibrated(l1)) == isclasscalibrated(l2)
end

compare_losses(PoissonLoss(), 2 * PoissonLoss())
compare_losses(PoissonLoss(), 0.5 * PoissonLoss())

@testset "Scaled losses" begin
  for loss in distance_losses ∪ margin_losses ∪ other_losses
    @testset "$loss" begin
      compare_losses(loss, 2 * loss)
      compare_losses(loss, 0.5 * loss)
    end
  end
end

@testset "Weighted Margin-based" begin
  for loss in margin_losses
    @testset "$loss" begin
      compare_losses(loss, WeightedMarginLoss(loss, 0.2), false)
      compare_losses(loss, WeightedMarginLoss(loss, 0.5), true)
      compare_losses(loss, WeightedMarginLoss(loss, 0.7), false)
    end
  end
end
