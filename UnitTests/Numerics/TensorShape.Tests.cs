using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Numerics.UnitTests;

public sealed partial class TensorShapeTests
{
    [Test]
    public async ValueTask Create_WithPositiveLengths_ShouldSuccess()
    {
        int[] lengths = [2, 3, 5];
        int[] strides = [15, 5, 1];
        int elementCount = 2 * 3 * 5;

        var shape = TensorShape.Create(lengths);

        await Assert.That(shape.Rank).IsEqualTo(lengths.Length);
        await Assert.That(shape.ElementCount).IsEqualTo(elementCount);

        await Assert.That(shape.Lengths.Length).IsEqualTo(lengths.Length);
        await Assert.That(shape.Strides.Length).IsEqualTo(strides.Length);

        for (int index = 0; index < lengths.Length; ++index)
        {
            await Assert.That(shape.Lengths[index]).IsEqualTo(lengths[index]);
            await Assert.That(shape.Strides[index]).IsEqualTo(strides[index]);
        }
    }

    [Test]
    public async ValueTask Create_WithNoLengths_ShouldReturnScalar()
    {
        var shape = TensorShape.Create();

        await Assert.That(shape.Rank).IsZero();
        await Assert.That(shape.ElementCount).IsEqualTo(1);
        await Assert.That(shape.Lengths.IsEmpty).IsTrue();
        await Assert.That(shape.Strides.IsEmpty).IsTrue();
    }

    [Test]
    public async ValueTask Create_WithAnyZeroLength_ShouldSetElementCountAndStridesAsZero()
    {
        int[] lengths = [2, 0, 5];

        var shape = TensorShape.Create(lengths);

        await Assert.That(shape.Rank).IsEqualTo(lengths.Length);
        await Assert.That(shape.ElementCount).IsZero();

        await Assert.That(shape.Lengths.Length).IsEqualTo(lengths.Length);
        await Assert.That(shape.Strides.Length).IsEqualTo(lengths.Length);

        for (int index = 0; index < shape.Rank; ++index)
        {
            await Assert.That(shape.Lengths[index]).IsEqualTo(lengths[index]);
            await Assert.That(shape.Strides[index]).IsZero();
        }
    }

    [Test]
    public async ValueTask Create_WithAnyNegativeLength_ShouldThrow()
    {
        int[] lengths = [2, -1, 5];

        var unit = () => TensorShape.Create(lengths);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask Create_SubShape_WithValidRanges_ShouldSuccess()
    {
        int[] lengths = [2, 3, 5];
        Range[] ranges = [0..1, 1..3, 2..5];
        int[] subLengths = [1, 2, 3];
        int elementCount = 6;
        int offset = 7;

        var shape = TensorShape.Create(lengths);

        var subShape = TensorShape.Create(shape, ranges, out int linearOffset);

        await Assert.That(subShape.Rank).IsEqualTo(lengths.Length);
        await Assert.That(subShape.ElementCount).IsEqualTo(elementCount);
        await Assert.That(subShape.Lengths.Length).IsEqualTo(ranges.Length);
        await Assert.That(subShape.Strides.Length).IsEqualTo(ranges.Length);

        for (int index = 0; index < subShape.Rank; ++index)
        {
            await Assert.That(subShape.Lengths[index]).IsEqualTo(subLengths[index]);
            await Assert.That(subShape.Strides[index]).IsEqualTo(shape.Strides[index]);
        }

        await Assert.That(linearOffset).IsEqualTo(offset);
    }

    [Test]
    public async ValueTask Create_SubShape_WithInvalidRanges_ShouldThrow()
    {
        int[] lengths = [2, 3, 5];
        Range[] ranges = [0..4, 1..3, 2..5];

        var shape = TensorShape.Create(lengths);

        var unit = () => TensorShape.Create(shape, ranges, out _);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask Create_SubShape_WithMissingRanges_ShouldThrow()
    {
        int[] lengths = [2, 3, 5];
        Range[] ranges = [0..1, 1..3];

        var shape = TensorShape.Create(lengths);

        var unit = () => TensorShape.Create(shape, ranges, out _);

        await Assert.That(unit).ThrowsException();
    }

    [Test]
    public async ValueTask Equals_WithSameShapes_ShouldReturnTrue()
    {
        int[] lengths = [2, 3, 5];

        var shape1 = TensorShape.Create(lengths);
        var shape2 = TensorShape.Create(lengths);

        await Assert.That(shape1.Equals(shape2)).IsTrue();
    }

    [Test]
    public async ValueTask Equals_WithDifferentShapes_ShouldReturnFalse()
    {
        int[] lengths1 = [2, 3, 5];
        int[] lengths2 = [2, 3, 1];

        var shape1 = TensorShape.Create(lengths1);
        var shape2 = TensorShape.Create(lengths2);

        await Assert.That(shape1.Equals(shape2)).IsFalse();
    }
}
