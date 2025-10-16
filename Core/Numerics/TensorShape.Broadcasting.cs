using System.Buffers;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Numerics;

public readonly partial struct TensorShape
{
    /// <summary>
    /// Tries to broadcast the specified shapes into a new one.
    /// </summary>
    /// <param name="shape1">The first shape to broadcast.</param>
    /// <param name="shape2">The second shape to broadcast.</param>
    /// <param name="result">When this method returns, contains the broadcasted shape.</param>
    /// <returns><see langword="true"/> if broadcasting was successful, <see langword="false"/> otherwise.</returns>
    public static bool TryBroadcast(in TensorShape shape1, in TensorShape shape2, out TensorShape result)
    {
        int rank1 = shape1.Rank;
        ReadOnlySpan<int> lengths1 = shape1.Lengths;

        int rank2 = shape2.Rank;
        ReadOnlySpan<int> lengths2 = shape2.Lengths;

        int broadcastRank = int.Max(rank1, rank2);

        int[] buffer = ArrayPool<int>.Shared.Rent(broadcastRank * 2);
        Span<int> broadcastLengths = buffer.AsSpan(broadcastRank * 0, broadcastRank);
        Span<int> broadcastStrides = buffer.AsSpan(broadcastRank * 1, broadcastRank);

        try
        {
            int elementCount = 1;

            for (int dimension = 1; dimension <= broadcastRank; ++dimension)
            {
                int length1 = (dimension <= rank1) ? lengths1[rank1 - dimension] : 1;
                int length2 = (dimension <= rank2) ? lengths2[rank2 - dimension] : 1;

                if (length1 != length2 && length1 != 1 && length2 != 1)
                {
                    result = TensorShape.Empty;
                    return false;
                }

                int broadcastLength = int.Max(length1, length2);
                broadcastLengths[broadcastRank - dimension] = broadcastLength;
                broadcastStrides[broadcastRank - dimension] = elementCount;
                elementCount = checked(elementCount * broadcastLength);
            }

            result = new TensorShape(broadcastRank, elementCount, broadcastLengths, broadcastStrides);
            return true;
        }
        finally
        {
            ArrayPool<int>.Shared.Return(buffer);
        }
    }

    /// <summary>
    /// Tries to broadcast this shape and <paramref name="other"/> into a new one.
    /// </summary>
    /// <param name="other">The other shape to broadcast.</param>
    /// <inheritdoc cref="TensorShape.TryBroadcast(in TensorShape, in TensorShape, out TensorShape)" path="/param[@name='result']"/>
    /// <inheritdoc cref="TensorShape.TryBroadcast(in TensorShape, in TensorShape, out TensorShape)" path="/returns"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly bool TryBroadcast(in TensorShape other, out TensorShape result)
    {
        return TensorShape.TryBroadcast(this, other, out result);
    }

    /// <summary>
    /// Checkes whether current shape is broadcastable to the <paramref name="other"/>.
    /// </summary>
    /// <param name="other">The other shape.</param>
    /// <returns><see langword="true"/> if current shape was broadcastable to other one; <see langword="false"/> otherwise.</returns>
    public readonly bool IsBroadcastableTo(in TensorShape other)
    {
        int rank = Rank;
        int broadcastRank = other.Rank;
        int rankDelta = broadcastRank - rank;

        if (rankDelta < 0)
        {
            return false;
        }

        ReadOnlySpan<int> lengths = Lengths;
        ReadOnlySpan<int> broadcastLengths = other.Lengths;

        for (int dimension = rank - 1; dimension >= 0; --dimension)
        {
            int length = lengths[dimension];
            int broadcastLength = broadcastLengths[dimension + rankDelta];

            if (length != broadcastLength && length != 1)
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Computes the linear indices of shapes respective to the specified <paramref name="indices"/> for the broadcasted shape.
    /// </summary>
    /// <param name="shape1">The first shape.</param>
    /// <param name="shape2">The second shape.</param>
    /// <param name="indices">The multi-dimensional indices for the broadcasted shape.</param>
    /// <returns>A tuple containing the linear index for <paramref name="shape1"/> and <paramref name="shape2"/>.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public static (int Index1, int Index2) ComputeBroadcastIndices(in TensorShape shape1, in TensorShape shape2, ReadOnlySpan<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(indices.Length, shape1.Rank);
        ArgumentOutOfRangeException.ThrowIfLessThan(indices.Length, shape2.Rank);

        int rank = indices.Length;

        ReadOnlySpan<int> lengths1 = shape1.Lengths;
        ReadOnlySpan<int> strides1 = shape1.Strides;

        ReadOnlySpan<int> lengths2 = shape2.Lengths;
        ReadOnlySpan<int> strides2 = shape2.Strides;

        int offset1 = rank - shape1.Rank;
        int offset2 = rank - shape2.Rank;

        int broadcastIndex1 = 0;
        int broadcastIndex2 = 0;

        for (int dimension = 0; dimension < rank; ++dimension)
        {
            int dimension1 = dimension - offset1;
            int dimension2 = dimension - offset2;

            int length1 = (dimension1 < 0) ? 1 : lengths1[dimension1];
            int length2 = (dimension2 < 0) ? 1 : lengths2[dimension2];

            int stride1 = (length1 == 1) ? 0 : strides1[dimension1];
            int stride2 = (length2 == 1) ? 0 : strides2[dimension2];

            int index = indices[dimension];
            broadcastIndex1 = checked(broadcastIndex1 + stride1 * index);
            broadcastIndex2 = checked(broadcastIndex2 + stride2 * index);
        }

        return (broadcastIndex1, broadcastIndex2);
    }

    /// <summary>
    /// Computes the linear index respective to the specified <paramref name="indices"/> for the broadcasted shape.
    /// </summary>
    /// <inheritdoc cref="ComputeBroadcastIndices(in TensorShape, in TensorShape, ReadOnlySpan{int})" path="/param[@name='indices']"/>
    /// <returns>A single integer representing the respective linear index.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public readonly int ComputeBroadcastIndex(ReadOnlySpan<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(indices.Length, Rank);

        ReadOnlySpan<int> lengths = Lengths;
        ReadOnlySpan<int> strides = Strides;

        int broadcastIndex = 0;
        int offset = indices.Length - Rank;

        for (int dimension = Rank - 1; dimension >= 0; --dimension)
        {
            int length = lengths[dimension];
            int stride = (length == 1) ? 0 : strides[dimension];

            int index = indices[dimension + offset];
            broadcastIndex = checked(broadcastIndex + stride * index);
        }

        return broadcastIndex;
    }
}
