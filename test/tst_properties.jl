immutable TstVanillaLoss <: SupervisedLoss end

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


immutable TstStronglyConvexLoss <: SupervisedLoss end
Losses.isstronglyconvex(::TstStronglyConvexLoss) = true

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


immutable TstTwiceDiffLoss <: SupervisedLoss end
Losses.istwicedifferentiable(::TstTwiceDiffLoss) = true

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


immutable TstMarginLoss <: MarginLoss end

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


immutable TstDistanceLoss <: DistanceLoss end

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

