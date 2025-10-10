using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Numerics.UnitTests;

public sealed partial class TensorShapeTests
{
    [Test]
    public async ValueTask InitializeForwardIndexing_WithNonEmptyIndices_ShouldSetIndices()
    {
        int[] indices = [-5, 17, -21];

        TensorShape.InitializeForwardIndexing(indices);

        int zeroCount = indices.AsSpan().Count(0);
        int lastElement = indices[^1];

        await Assert.That(zeroCount).IsEquivalentTo(indices.Length - 1);
        await Assert.That(lastElement).IsEqualTo(-1);
    }

    [Test]
    public async ValueTask InitializeForwardIndexing_WithEmptyIndices_ShouldThrow()
    {
        var unit = () => TensorShape.InitializeForwardIndexing([]);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask InitializeBackwardIndexing_WithNonEmptyIndices_ShouldSetIndices()
    {
        int[] lengths = [2, 3, 5];
        int[] indices = [-5, 17, -21];

        TensorShape.InitializeBackwardIndexing(lengths, indices);

        for (int i = 0; i < indices.Length - 1; ++i)
        {
            await Assert.That(indices[i]).IsEqualTo(lengths[i] - 1);
        }

        await Assert.That(indices[^1]).IsEqualTo(lengths[^1]);
    }

    [Test]
    public async ValueTask InitializeBackwardIndexing_WithMismatchedLength_ShouldThrow()
    {
        int[] lengths = [2, 3, 5];
        int[] indices = [-5, 17];

        var unit = () => TensorShape.InitializeBackwardIndexing(lengths, indices);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask MoveToNextIndex_WithNonEmptyIndices_ShouldIterateAllElements()
    {
        int[] indices = [0, 0, -1];
        var shape = TensorShape.Create(2, 3, 5);

        int iterationCount = 0;
        while (shape.MoveToNextIndex(indices))
        {
            ++iterationCount;
        }

        await Assert.That(iterationCount).IsEqualTo(shape.ElementCount);
    }

    [Test]
    public async ValueTask MoveToNextIndex_WithMoreIndicesThanRank_ShouldThrow()
    {
        int[] indices = [0, 0, 0, -1];
        var shape = TensorShape.Create(2, 3, 5);

        var unit = () => shape.MoveToNextIndex(indices);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask MoveToPreviousIndex_WithNonEmptyIndices_ShouldIterateAllElements()
    {
        int[] indices = [1, 2, 5];
        var shape = TensorShape.Create(2, 3, 5);

        int iterationCount = 0;
        while (shape.MoveToPreviousIndex(indices))
        {
            ++iterationCount;
        }

        await Assert.That(iterationCount).IsEqualTo(shape.ElementCount);
    }

    [Test]
    public async ValueTask MoveToPreviousIndex_WithMoreIndicesThanRank_ShouldThrow()
    {
        int[] indices = [0, 1, 2, 5];
        var shape = TensorShape.Create(2, 3, 5);

        var unit = () => shape.MoveToPreviousIndex(indices);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask ComputeLinearIndex_WithValidIndices_OfLengthEqualToRank_ShouldComputeCorrectly()
    {
        int[] indices = [1, 2, 4];
        var shape = TensorShape.Create(2, 3, 5);

        int linearIndex = shape.ComputeLinearIndex(indices);

        await Assert.That(linearIndex).IsEqualTo(shape.ElementCount - 1);
    }

    [Test]
    public async ValueTask ComputeLinearIndex_WithValidIndices_OfLengthLessThanRank_ShouldComputeCorrectly()
    {
        int[] indices = [5, 4];
        var shape = TensorShape.Create(2, 3, 5);

        int linearIndex = shape.ComputeLinearIndex(indices);

        await Assert.That(linearIndex).IsEqualTo(shape.ElementCount - 1);
    }

    [Test]
    public async ValueTask ComputeLinearIndex_WithOutOfRangeIndices_ShouldThrow()
    {
        int[] indices = [7, 2, 4];
        var shape = TensorShape.Create(2, 3, 5);

        var unit = () => shape.ComputeLinearIndex(indices);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask ComputeLinearIndex_WithNegativeIndices_ShouldThrow()
    {
        int[] indices = [-7, 2, 4];
        var shape = TensorShape.Create(2, 3, 5);

        var unit = () => shape.ComputeLinearIndex(indices);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask ComputeLinearIndexUnchecked_InAnyCase_ShouldComputeCorrectly()
    {
        int[] indices = [1, 2, 4];
        var shape = TensorShape.Create(2, 3, 5);

        int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

        await Assert.That(linearIndex).IsEqualTo(shape.ElementCount - 1);
    }
}
