namespace NeuralNetworks.Core.Numerics;

public readonly partial struct TensorShape
{
    /// <summary>
    /// Initializes the <paramref name="indices"/> for a forward iteration.
    /// </summary>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <remarks>The length of the <paramref name="indices"/> should be same as rank of the shape.</remarks>
    public static void InitializeForwardIndexing(Span<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfZero(indices.Length);

        indices.Clear();
        indices[^1] = -1;
    }

    /// <summary>
    /// Initializes the <paramref name="indices"/> for a backward iteration.
    /// </summary>
    /// <param name="lengths">The lengths of the elements.</param>
    /// <inheritdoc cref="InitializeForwardIndexing(Span{int})" path="/param[@name='indices']"/>
    /// <inheritdoc cref="InitializeForwardIndexing(Span{int})" path="/remarks"/>
    public static void InitializeBackwardIndexing(ReadOnlySpan<int> lengths, Span<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(indices.Length, lengths.Length);

        lengths.CopyTo(indices);
        for (int dimension = lengths.Length - 2; dimension >= 0; --dimension)
        {
            --indices[dimension];
        }
    }

    /// <summary>
    /// Moves to the next linear index by incrementing the <paramref name="indices"/> according to the specified <paramref name="lengths"/>.
    /// </summary>
    /// <param name="lengths">The lengths in each dimension.</param>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <returns><see langword="true"/> if there was a next index; <see langword="false"/> otherwise.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public static bool MoveToNextIndex(ReadOnlySpan<int> lengths, Span<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(indices.Length, lengths.Length);

        int rank = indices.Length;
        int offset = lengths.Length - rank;

        for (int dimension = rank - 1; dimension >= 0; --dimension)
        {
            int length = lengths[dimension + offset];
            ref int index = ref indices[dimension];

            if (++index < length)
            {
                return true;
            }

            index = 0;
        }

        return false;
    }

    /// <summary>
    /// Moves to the next linear index by incrementing the <paramref name="indices"/> according to the <see cref="TensorShape.Lengths"/>.
    /// </summary>
    /// <inheritdoc cref="MoveToNextIndex(ReadOnlySpan{int}, Span{int})"/>
    public readonly bool MoveToNextIndex(Span<int> indices)
    {
        return TensorShape.MoveToNextIndex(Lengths, indices);
    }

    /// <summary>
    /// Moves to the next linear index by decrementing the <paramref name="indices"/> according to the specified <paramref name="lengths"/>.
    /// </summary>
    /// <inheritdoc cref="MoveToNextIndex(ReadOnlySpan{int}, Span{int})" path="/param"/>
    /// <returns><see langword="true"/> if there was a previous index; <see langword="false"/> otherwise.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public static bool MoveToPreviousIndex(ReadOnlySpan<int> lengths, Span<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfGreaterThan(indices.Length, lengths.Length);

        int rank = indices.Length;
        int offset = lengths.Length - rank;

        for (int dimension = rank - 1; dimension >= 0; --dimension)
        {
            int length = lengths[dimension + offset];
            ref int index = ref indices[dimension];

            if (--index >= 0)
            {
                return true;
            }

            index = length - 1;
        }

        return false;
    }

    /// <summary>
    /// Moves to the next linear index by decrementing the <paramref name="indices"/> according to the <see cref="TensorShape.Lengths"/>.
    /// </summary>
    /// <inheritdoc cref="MoveToPreviousIndex(ReadOnlySpan{int}, Span{int})"/>
    public readonly bool MoveToPreviousIndex(Span<int> indices)
    {
        return TensorShape.MoveToPreviousIndex(Lengths, indices);
    }

    /// <summary>
    /// Computes a linear index using the specified multi-dimensional <paramref name="indices"/>.
    /// </summary>
    /// <param name="indices">The multi-dimensional indices.</param>
    /// <returns>A single integer representing the linear index.</returns>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public readonly int ComputeLinearIndex(params ReadOnlySpan<int> indices)
    {
        ArgumentOutOfRangeException.ThrowIfZero(indices.Length);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(indices.Length, Rank);

        ReadOnlySpan<int> lengths = Lengths[^indices.Length..];
        ReadOnlySpan<int> strides = Strides[^indices.Length..];

        int linearIndex = 0;

        for (int dimension = indices.Length - 1; dimension > 0; --dimension)
        {
            int length = lengths[dimension];
            int stride = strides[dimension];
            int index = indices[dimension];

            ArgumentOutOfRangeException.ThrowIfNegative(index);
            ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(index, length);

            linearIndex = checked(linearIndex + index * stride);
        }

        linearIndex = checked(linearIndex + indices[0] * strides[0]);
        if (linearIndex >= ElementCount)
        {
            throw new ArgumentException($"{nameof(indices)} points to out of range of elements.");
        }

        return linearIndex;
    }

    /// <inheritdoc cref="ComputeLinearIndex(ReadOnlySpan{int})"/>
    /// <remarks>
    /// This method does not checks for bounds for performance, so use it with caution.
    /// </remarks>
    public readonly int ComputeLinearIndexUnchecked(scoped ReadOnlySpan<int> indices)
    {
        ReadOnlySpan<int> strides = Strides[^indices.Length..];

        int linearIndex = 0;

        for (int dimension = indices.Length - 1; dimension >= 0; --dimension)
        {
            linearIndex += indices[dimension] * strides[dimension];
        }

        return linearIndex;
    }
}
