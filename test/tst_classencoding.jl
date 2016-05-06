using MLBase

@testset "BinaryClassEncoding: class hierachy" begin
    @test BinaryClassEncoding <: ClassEncoding
    @test ZeroOneClassEncoding <: BinaryClassEncoding
    @test SignedClassEncoding <: BinaryClassEncoding
    @test ClassEncodings.ZeroOneClassEncoding == ZeroOneClassEncoding
    @test ClassEncodings.SignedClassEncoding == SignedClassEncoding
end

@testset "BinaryClassEncoding: interface stability" begin
    wrongDim1 = ["y","y","y"]
    wrongDim2 = ["y","n","c"]

    @test_throws ArgumentError ZeroOneClassEncoding(wrongDim1)
    @test_throws ArgumentError ZeroOneClassEncoding(wrongDim2)
    @test_throws ArgumentError SignedClassEncoding(wrongDim1)
    @test_throws ArgumentError SignedClassEncoding(wrongDim2)

    t = ["y", "n", "n", "y", "n"]
    wrongLabel1 = ["a","b","a"]
    wrongLabel2 = [1,2,2]

    ce = ZeroOneClassEncoding(t)
    @test_throws KeyError labelencode(ce, wrongLabel1)
    @test_throws MethodError labelencode(ce, wrongLabel2)

    ce = SignedClassEncoding(t)
    @test_throws KeyError labelencode(ce, wrongLabel1)
    @test_throws MethodError labelencode(ce, wrongLabel2)
end

@testset "BinaryClassEncoding: encoding and decoding" begin
    t = ["y", "n", "n", "y", "n"]
    y = ["n","y", "y", "n"]

    ce = ZeroOneClassEncoding(t)
    pred = labelencode(ce, y)
    idx = groupindices(ce, t)
    @test idx[1] == [1,4]
    @test idx[2] == [2,3,5]
    @test classdistribution(ce, t) == (["y","n"], [2, 3])
    @test classdistribution(ce.labelmap, t) == (["y","n"], [2, 3])
    @test pred == [1., 0, 0, 1]
    @test labeldecode(ce, pred) == y

    ce = ZeroOneClassEncoding(labelmap(["F","T"]))
    @test labeldecode(ce, [1., 0, 1]) == ["T", "F", "T"]
    @test labeldecode(ce, [0., 1, 0]) == ["F", "T", "F"]

    ce = SignedClassEncoding(t)
    pred = labelencode(ce, y)
    idx = groupindices(ce, t)
    @test idx[1] == [1,4]
    @test idx[2] == [2,3,5]
    @test classdistribution(ce, t) == (["y","n"], [2, 3])
    @test classdistribution(ce.labelmap, t) == (["y","n"], [2, 3])
    @test pred == [1., -1, -1, 1]
    @test labeldecode(ce, pred) == y

    ce = SignedClassEncoding(labelmap(["F","T"]))
    @test labeldecode(ce, [1., -1, 1]) == ["T", "F", "T"]
    @test labeldecode(ce, [-1., 1, -1]) == ["F", "T", "F"]
end

@testset "MultinomialClassEncoding: class hierachy" begin
    @test MultinomialClassEncoding <: ClassEncoding
    @test MultivalueClassEncoding <: MultinomialClassEncoding
    @test OneOfKClassEncoding <: MultinomialClassEncoding
    @test OneHotClassEncoding <: MultinomialClassEncoding
    @test OneHotClassEncoding == OneOfKClassEncoding
    @test ClassEncodings.MultivalueClassEncoding == MultivalueClassEncoding
    @test ClassEncodings.OneOfKClassEncoding == OneOfKClassEncoding
end

@testset "MultinomialClassEncoding: interface stability" begin
    wrongDim1 = ["y","y","y"]

    @test_throws ArgumentError MultivalueClassEncoding(wrongDim1)
    @test_throws ArgumentError OneOfKClassEncoding(wrongDim1)

    t = ["y", "n", "k", "y", "n"]
    wrongLabel1 = ["a","b","a"]
    wrongLabel2 = [1,2,2]

    ce = MultivalueClassEncoding(t)
    @test_throws KeyError labelencode(ce, wrongLabel1)
    @test_throws MethodError labelencode(ce, wrongLabel2)

    ce = OneOfKClassEncoding(t)
    @test_throws KeyError labelencode(ce, wrongLabel1)
    @test_throws MethodError labelencode(ce, wrongLabel2)
end

@testset "MultinomialClassEncoding: encoding and decoding" begin
    t = ["y", "n", "k", "y", "n"]
    y = ["n", "y", "y", "k", "n"]

    ce = MultivalueClassEncoding(t)
    pred = labelencode(ce, y)
    idx = groupindices(ce, t)
    @test idx[1] == [1,4]
    @test idx[2] == [2,5]
    @test idx[3] == [3]
    @test classdistribution(ce, t) == (["y","n","k"], [2, 2, 1])
    @test classdistribution(ce.labelmap, t) == (["y","n","k"], [2, 2, 1])
    @test pred == [2., 1, 1, 3, 2]
    @test labeldecode(ce, pred) == y

    ce = MultivalueClassEncoding(t, zero_based=true)
    pred = labelencode(ce, y)
    idx = groupindices(ce, t)
    @test idx[1] == [1,4]
    @test idx[2] == [2,5]
    @test idx[3] == [3]
    @test classdistribution(ce, t) == (["y","n","k"], [2, 2, 1])
    @test classdistribution(ce.labelmap, t) == (["y","n","k"], [2, 2, 1])
    @test pred == [1., 0, 0, 2, 1]
    @test labeldecode(ce, pred) == y

    ce = MultivalueClassEncoding(labelmap(["F","T", "W"]))
    @test labeldecode(ce, [2., 1, 3]) == ["T", "F", "W"]
    @test labeldecode(ce, [3., 2, 1]) == ["W", "T", "F"]

    ce = OneOfKClassEncoding(t)
    pred = labelencode(ce, y)
    @test pred ==
    [0.  1  0;
    1  0  0;
    1  0  0;
    0  0  1;
    0  1  0]'
    @test labeldecode(ce, pred) == y
end

