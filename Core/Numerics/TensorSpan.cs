using System.Numerics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Numerics;

/// <summary>
/// Represents span of a multi-dimensional numerical data.
/// </summary>
/// <inheritdoc cref="ITensor{TScalar}" path="/typeparam"/>
public readonly ref struct TensorSpan<TScalar> : ITensor<TScalar>
    where TScalar : struct, INumberBase<TScalar>
{
    /// <summary>Gets the empty instance.</summary>
    public static TensorSpan<TScalar> Empty => default;

    private readonly TensorShape _shape;
    private readonly ref TScalar _address;

    /// <inheritdoc/>
    public readonly TensorShape Shape => _shape;

    /// <inheritdoc/>
    public readonly bool IsEmpty => _shape.ElementCount == 0;

    /// <inheritdoc/>
    public readonly bool IsScalar => _shape.ElementCount == 1;

    /// <inheritdoc/>
    public readonly ref TScalar this[params ReadOnlySpan<int> indices] => ref ElementAt(_shape.ComputeLinearIndex(indices));

    /// <inheritdoc/>
    public readonly TensorSpan<TScalar> this[params ReadOnlySpan<Range> ranges] => Slice(ranges);

    internal TensorSpan(TensorShape shape, ref TScalar address)
    {
        _shape = shape;
        _address = ref address;
    }

    internal readonly ref TScalar ElementAt(int linearIndex)
    {
        return ref Unsafe.Add(ref _address, linearIndex);
    }

    /// <summary>
    /// Forms a slice out of the current span within the specified <paramref name="ranges"/>
    /// </summary>
    /// <param name="ranges">The multi-dimensional ranges.</param>
    /// <returns>A new instance of <see cref="TensorSpan{TScalar}"/>.</returns>
    public TensorSpan<TScalar> Slice(scoped ReadOnlySpan<Range> ranges)
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
