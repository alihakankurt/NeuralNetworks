using System.Buffers;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Numerics;

/// <summary>
/// Represents shape of a tensor.
/// </summary>
public readonly partial struct TensorShape : IEquatable<TensorShape>
{
    /// <summary>The maximum value of rank, to store without memory allocation.</summary>
    public const int MaxInlineRank = 5;

    /// <summary>Gets the empty instance.</summary>
    public static TensorShape Empty => default;

    private readonly int[]? _dimensionData;
    private readonly InlineDimensionData _inlineLengths;
    private readonly InlineDimensionData _inlineStrides;

    /// <summary>Gets the number of dimensions.</summary>
    public readonly int Rank { get; }

    /// <summary>Gets the total number of elements.</summary>
    public readonly int ElementCount { get; }

    /// <summary>Gets the length in each dimension.</summary>
    [UnscopedRef]
    public readonly ReadOnlySpan<int> Lengths
    {
        get
        {
            if (Rank <= MaxInlineRank)
            {
                return _inlineLengths[..Rank];
            }
            else
            {
                return _dimensionData.AsSpan(Rank * 0, Rank);
            }
        }
    }

    /// <summary>Gets the element-wise stride in each dimension.</summary>
    [UnscopedRef]
    public readonly ReadOnlySpan<int> Strides
    {
        get
        {
            if (Rank <= MaxInlineRank)
            {
                return _inlineStrides[..Rank];
            }
            else
            {
                return _dimensionData.AsSpan(Rank * 1, Rank);
            }
        }
    }

    /// <summary>
    /// Initializes a new instance of <see cref="TensorShape"/> with specified arguments.
    /// </summary>
    private TensorShape(int rank, int elementCount, ReadOnlySpan<int> lengths, ReadOnlySpan<int> strides)
    {
        scoped Span<int> destinationLengths;
        scoped Span<int> destinationStrides;

        if (rank <= MaxInlineRank)
        {
            Unsafe.SkipInit(out _inlineLengths);
            Unsafe.SkipInit(out _inlineStrides);

            destinationLengths = _inlineLengths[..rank];
            destinationStrides = _inlineStrides[..rank];
        }
        else
        {
            _dimensionData = GC.AllocateUninitializedArray<int>(2 * rank);

            destinationLengths = _dimensionData.AsSpan(rank * 0, rank);
            destinationStrides = _dimensionData.AsSpan(rank * 1, rank);
        }

        Rank = rank;
        ElementCount = elementCount;
        lengths.CopyTo(destinationLengths);

        if (elementCount == 0)
        {
            destinationStrides.Clear();
        }
        else
        {
            strides.CopyTo(destinationStrides);
        }
    }

    /// <summary>
    /// Creates a new <see cref="TensorShape"/> with the specified <paramref name="lengths"/>.
    /// </summary>
    /// <param name="lengths">The lengths in each dimension.</param>
    /// <returns>A new instance of <see cref="TensorShape"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    /// <remarks>
    /// If the <see cref="Rank"/> is less than or equal to the <see cref="MaxInlineRank"/>, no additional memory allocation will be required.
    /// </remarks>
    public static TensorShape Create(params ReadOnlySpan<int> lengths)
    {
        int rank = lengths.Length;
        if (rank == 0)
        {
            return new TensorShape(0, 1, [], []);
        }

        int[] buffer = ArrayPool<int>.Shared.Rent(rank);
        Span<int> strides = buffer.AsSpan(0, rank);

        try
        {
            int elementCount = 1;

            for (int dimension = rank - 1; dimension >= 0; --dimension)
            {
                int length = lengths[dimension];
                ArgumentOutOfRangeException.ThrowIfNegative(length);

                strides[dimension] = elementCount;
                elementCount = checked(elementCount * length);
            }

            return new TensorShape(rank, elementCount, lengths, strides);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Creates a sub-shape of the specified <paramref name="shape"/> within the specified <paramref name="ranges"/>.
    /// </summary>
    /// <param name="shape">The original shape.</param>
    /// <param name="ranges">The ranges in each dimension.</param>
    /// <param name="linearOffset">When this method returns, contains the linear offset to the first element.</param>
    /// <returns>A new instance of <see cref="TensorShape"/> as sub-shape.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    /// <inheritdoc cref="TensorShape.Create(ReadOnlySpan{int})" path="/remarks"/>
    public static TensorShape Create(in TensorShape shape, ReadOnlySpan<Range> ranges, out int linearOffset)
    {
        ArgumentOutOfRangeException.ThrowIfZero(ranges.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(ranges.Length, shape.Rank);

        int rank = ranges.Length;
        ReadOnlySpan<int> lengths = shape.Lengths;
        ReadOnlySpan<int> strides = shape.Strides;

        int[] buffer = ArrayPool<int>.Shared.Rent(rank);
        Span<int> subLengths = buffer.AsSpan(0, rank);

        try
        {
            linearOffset = 0;
            int elementCount = 1;

            for (int dimension = rank - 1; dimension >= 0; --dimension)
            {
                int length = lengths[dimension];
                int stride = strides[dimension];

                Range range = ranges[dimension];
                var (offset, subLength) = range.GetOffsetAndLength(length);

                subLengths[dimension] = subLength;
                elementCount = checked(elementCount * subLength);
                linearOffset = checked(linearOffset + offset * stride);
            }

            return new TensorShape(rank, elementCount, subLengths, shape.Strides);
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Creates a sub-shape from this one within the specified <paramref name="ranges"/>.
    /// </summary>
    /// <inheritdoc cref="TensorShape.Create(in TensorShape, ReadOnlySpan{Range}, out int)" path="/param"/>
    /// <inheritdoc cref="TensorShape.Create(in TensorShape, ReadOnlySpan{Range}, out int)" path="/returns"/>
    /// <inheritdoc cref="TensorShape.Create(in TensorShape, ReadOnlySpan{Range}, out int)" path="/exception"/>
    public readonly TensorShape SubShape(ReadOnlySpan<Range> ranges, out int linearOffset)
    {
        return TensorShape.Create(this, ranges, out linearOffset);
    }

    /// <summary>
    /// Checks whether the shape has equal rank and lengths with the <paramref name="other"/>.
    /// </summary>
    /// <param name="other">The other shape to compare.</param>
    /// <returns><see langword="true"/> if the shapes were equal; <see langword="false"/> otherwise.</returns>
    public readonly bool Equals(TensorShape other)
    {
        return (Rank == other.Rank) && Lengths.SequenceEqual(other.Lengths);
    }

    /// <inheritdoc />
    public readonly override bool Equals([NotNullWhen(true)] object? obj)
    {
        return obj is TensorShape shape && Equals(shape);
    }

    /// <inheritdoc />
    public readonly override int GetHashCode()
    {
        var instance = new HashCode();
        instance.Add(Rank);
        instance.AddBytes(MemoryMarshal.Cast<int, byte>(Lengths));
        return instance.ToHashCode();
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator ==(in TensorShape left, in TensorShape right)
    {
        return left.Equals(right);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static bool operator !=(in TensorShape left, in TensorShape right)
    {
        return !(left == right);
    }

    /// <summary>
    /// Represents the stack-allocated storage of lengths and strides.
    /// </summary>
    [InlineArray(MaxInlineRank)]
    private struct InlineDimensionData
    {
        private int _e0;
    }
}
