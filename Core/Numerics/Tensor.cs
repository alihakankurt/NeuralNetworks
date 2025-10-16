using System.Numerics;

namespace NeuralNetworks.Core.Numerics;

/// <summary>
/// Represents a multi-dimensional numerical data in space.
/// </summary>
/// <inheritdoc cref="ITensor{TScalar}" path="/typeparam"/>
public sealed class Tensor<TScalar> : ITensor<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    private readonly TensorShape _shape;
    private readonly TScalar[] _storage;

    /// <inheritdoc/>
    public TensorShape Shape => _shape;

    /// <inheritdoc/>
    public bool IsEmpty => _shape.ElementCount == 0;

    /// <inheritdoc/>
    public bool IsScalar => _shape.ElementCount == 1;

    /// <summary>Gets a reference to the element where the <paramref name="linearIndex"/> points to.</summary>
    public ref TScalar this[int linearIndex] => ref ElementAt(linearIndex);

    /// <inheritdoc/>
    public ref TScalar this[params ReadOnlySpan<int> indices] => ref ElementAt(_shape.ComputeLinearIndexUnchecked(indices));

    /// <inheritdoc/>
    public TensorSpan<TScalar> this[params ReadOnlySpan<Range> ranges] => AsSpan(ranges);

    internal Tensor(TensorShape shape, TScalar[] storage)
    {
        _shape = shape;
        _storage = storage;
    }

    internal ref TScalar ElementAt(int linearIndex)
    {
        return ref _storage[linearIndex];
    }

    /// <summary>
    /// Creates a new span over the current one.
    /// </summary>
    /// <returns>A new instance of <see cref="TensorSpan{TScalar}"/>.</returns>
    public TensorSpan<TScalar> AsSpan()
    {
        if (IsEmpty)
        {
            return TensorSpan<TScalar>.Empty;
        }

        TensorShape shape = Shape;
        ref TScalar address = ref ElementAt(0);

        return new TensorSpan<TScalar>(shape, ref address);
    }

    /// <summary>
    /// Creates a new span over the current one within the specified <paramref name="ranges"/>.
    /// </summary>
    /// <param name="ranges">The multi-dimensional ranges.</param>
    /// <returns>A new instance of <see cref="TensorSpan{TScalar}"/>.</returns>
    public TensorSpan<TScalar> AsSpan(scoped ReadOnlySpan<Range> ranges)
    {
        if (IsEmpty)
        {
            return TensorSpan<TScalar>.Empty;
        }

        TensorShape shape = Shape.SubShape(ranges, out int elementOffset);
        ref TScalar address = ref ElementAt(elementOffset);

        return new TensorSpan<TScalar>(shape, ref address);
    }
}
