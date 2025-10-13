using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Numerics.UnitTests;

public sealed partial class TensorShapeTests
{
    [Test]
    public async ValueTask TryBroadcast_WithCompatibleShapes_ShouldSuccess()
    {
        var shape1 = TensorShape.Create(2, 3, 1);
        var shape2 = TensorShape.Create(1, 3, 5);
        int[] lengths = [2, 3, 5];
        int elementCount = 2 * 3 * 5;

        bool isBroadcasted = shape1.TryBroadcast(shape2, out TensorShape result);

        await Assert.That(isBroadcasted).IsTrue();
        await Assert.That(result.Rank).IsEqualTo(lengths.Length);
        await Assert.That(result.ElementCount).IsEqualTo(elementCount);

        for (int index = 0; index < lengths.Length; ++index)
        {
            await Assert.That(result.Lengths[index]).IsEqualTo(lengths[index]);
        }
    }

    [Test]
    public async ValueTask TryBroadcast_WithIncompatibleShapes_ShouldFail()
    {
        var shape1 = TensorShape.Create(2, 3, 7);
        var shape2 = TensorShape.Create(1, 3, 5);

        bool isBroadcasted = shape1.TryBroadcast(shape2, out _);

        await Assert.That(isBroadcasted).IsFalse();
    }

    [Test]
    public async ValueTask IsBroadcastableTo_WithCompatibleShapes_ShouldSuccess()
    {
        var shape1 = TensorShape.Create(2, 3, 5);
        var shape2 = TensorShape.Create(3, 1);

        bool isBroadcastable = shape2.IsBroadcastableTo(shape1);

        await Assert.That(isBroadcastable).IsTrue();
    }

    [Test]
    public async ValueTask IsBroadcastableTo_WithIncompatibleShapes_ShouldFail()
    {
        var shape1 = TensorShape.Create(2, 3, 7);
        var shape2 = TensorShape.Create(1, 3, 5);

        bool isBroadcastable = shape2.IsBroadcastableTo(shape1);

        await Assert.That(isBroadcastable).IsFalse();
    }

    [Test]
    public async ValueTask ComputeBroadcastIndices_WithShapesAndIndices_ShouldCompute()
    {
        var shape1 = TensorShape.Create(2, 3, 1);
        var shape2 = TensorShape.Create(1, 3, 5);

        int[] indices = [1, 2, 4];
        int broadcastIndex1 = 5;
        int broadcastIndex2 = 14;

        var (index1, index2) = TensorShape.ComputeBroadcastIndices(shape1, shape2, indices);

        await Assert.That(index1).IsEqualTo(broadcastIndex1);
        await Assert.That(index2).IsEqualTo(broadcastIndex2);
    }

    [Test]
    public async ValueTask ComputeBroadcastIndices_WithMissingIndex_ShouldThrow()
    {
        var shape1 = TensorShape.Create(2, 3, 1);
        var shape2 = TensorShape.Create(1, 3, 5);

        int[] indices = [1, 2];

        var unit = () => TensorShape.ComputeBroadcastIndices(shape1, shape2, indices);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask ComputeBroadcastIndex_WithValidIndices_ShouldCompute()
    {
        var shape = TensorShape.Create(2, 3, 1);

        int[] indices = [1, 2, 4];
        int broadcastIndex = 5;

        var index = shape.ComputeBroadcastIndex(indices);

        await Assert.That(index).IsEqualTo(broadcastIndex);
    }

    [Test]
    public async ValueTask ComputeBroadcastIndex_WithMissingIndex_ShouldThrow()
    {
        var shape = TensorShape.Create(2, 3, 1);

        int[] indices = [1, 2];

        var unit = () => shape.ComputeBroadcastIndex(indices);

        await Assert.That(unit).ThrowsException();
    }
}
