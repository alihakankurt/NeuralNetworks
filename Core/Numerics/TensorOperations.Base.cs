using System.Numerics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Numerics;

/// <summary>
/// Provides tensor operations.
/// </summary>
public static partial class Tensor
{
    /// <summary>
    /// Accumulates the values of the <see cref="tensor"/> by an operation.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="operation">The cumulative operation to execute.</param>
    /// <param name="initial">The initial accumulator value.</param>
    /// <returns>The accumulated value.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static TScalar Accumulate<TScalar>(this Tensor<TScalar> tensor, CumulativeOperation<TScalar> operation, TScalar initial)
        where TScalar : struct, INumberBase<TScalar>
    {
        return Tensor.Accumulate<TScalar>(tensor.AsSpan(), operation, initial);
    }

    /// <summary>
    /// Accumulates the values of the <see cref="span"/> by an operation.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="operation">The cumulative operation to execute.</param>
    /// <param name="initial">The initial accumulator value.</param>
    /// <returns>The accumulated value.</returns>
    public static TScalar Accumulate<TScalar>(this in TensorSpan<TScalar> span, CumulativeOperation<TScalar> operation, TScalar initial)
        where TScalar : struct, INumberBase<TScalar>
    {
        TensorShape shape = span.Shape;
        var result = Tensor.Create<TScalar>(shape);

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        TScalar accumulator = initial;

        while (shape.MoveToNextIndex(indices))
        {
            int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

            TScalar element = span.ElementAt(linearIndex);
            ref TScalar resultElement = ref result.ElementAt(linearIndex);

            accumulator = operation(accumulator, element);
        }

        return accumulator;
    }

    /// <summary>
    /// Executes an unary operation over the <see cref="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="operation">The unary operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> Execute<TScalar>(this Tensor<TScalar> tensor, UnaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        return Tensor.Execute<TScalar>(tensor.AsSpan(), operation);
    }

    /// <summary>
    /// Executes an unary operation over the <see cref="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="operation">The unary operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    public static Tensor<TScalar> Execute<TScalar>(this in TensorSpan<TScalar> span, UnaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        TensorShape shape = span.Shape;
        var result = Tensor.Create<TScalar>(shape);

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

            TScalar element = span.ElementAt(linearIndex);
            ref TScalar resultElement = ref result.ElementAt(linearIndex);

            resultElement = operation(element);
        }

        return result;
    }

    /// <summary>
    /// Executes an unary operation over the <see cref="tensor"/> in-place.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="operation">The unary operation to execute.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteIn<TScalar>(this Tensor<TScalar> tensor, UnaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        Tensor.ExecuteIn(tensor.AsSpan(), operation);
    }

    /// <summary>
    /// Executes an unary operation over the <see cref="span"/> in-place.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="operation">The unary operation to execute.</param>
    public static void ExecuteIn<TScalar>(this in TensorSpan<TScalar> span, UnaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        TensorShape shape = span.Shape;

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

            ref TScalar resultElement = ref span.ElementAt(linearIndex);

            resultElement = operation(resultElement);
        }
    }

    /// <summary>
    /// Executes a binary operation over <see cref="left"/> and <see cref="right"/> tensors.
    /// </summary>
    /// <param name="left">The left tensor.</param>
    /// <param name="right">The right tensor.</param>
    /// <param name="operation">The binary operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> Execute<TScalar>(this Tensor<TScalar> left, Tensor<TScalar> right, BinaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        return Tensor.Execute<TScalar>(left.AsSpan(), right.AsSpan(), operation);
    }

    /// <summary>
    /// Executes a binary operation over <see cref="left"/> and <see cref="right"/> spans.
    /// </summary>
    /// <param name="left">The left tensor span.</param>
    /// <param name="right">The right tensor span.</param>
    /// <param name="operation">The binary operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    public static Tensor<TScalar> Execute<TScalar>(this in TensorSpan<TScalar> left, in TensorSpan<TScalar> right, BinaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        TensorShape leftShape = left.Shape;
        TensorShape rightShape = right.Shape;

        if (!TensorShape.TryBroadcast(leftShape, rightShape, out var shape))
        {
            throw new InvalidOperationException($"The {nameof(left)} and {nameof(right)} has incompatible shapes to broadcast.");
        }

        var result = Tensor.Create<TScalar>(shape);

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            var (leftLinearIndex, rightLinearIndex) = TensorShape.ComputeBroadcastIndices(leftShape, rightShape, indices);
            var resultIndex = shape.ComputeLinearIndexUnchecked(indices);

            TScalar leftElement = left.ElementAt(leftLinearIndex);
            TScalar rightElement = right.ElementAt(rightLinearIndex);
            ref TScalar resultElement = ref result.ElementAt(resultIndex);

            resultElement = operation(leftElement, rightElement);
        }

        return result;
    }


    /// <summary>
    /// Executes a binary operation over <see cref="left"/> and <see cref="right"/> tensors in-place.
    /// </summary>
    /// <param name="left">The left tensor.</param>
    /// <param name="right">The right tensor.</param>
    /// <param name="operation">The binary operation to execute.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteIn<TScalar>(this Tensor<TScalar> left, Tensor<TScalar> right, BinaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        ExecuteIn(left.AsSpan(), right.AsSpan(), operation);
    }

    /// <summary>
    /// Executes a binary operation over <see cref="left"/> and <see cref="right"/> spans in-place.
    /// </summary>
    /// <param name="left">The left tensor span.</param>
    /// <param name="right">The right tensor span.</param>
    /// <param name="operation">The binary operation to execute.</param>
    public static void ExecuteIn<TScalar>(this in TensorSpan<TScalar> left, in TensorSpan<TScalar> right, BinaryOperation<TScalar> operation)
        where TScalar : struct, INumberBase<TScalar>
    {
        TensorShape leftShape = left.Shape;
        TensorShape rightShape = right.Shape;

        if (!rightShape.IsBroadcastableTo(leftShape))
        {
            throw new InvalidOperationException($"The {nameof(left)} and {nameof(right)} has incompatible shapes to execute in-place operation.");
        }

        Span<int> indices = stackalloc int[leftShape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (leftShape.MoveToNextIndex(indices))
        {
            var leftLinearIndex = leftShape.ComputeLinearIndexUnchecked(indices);
            var rightLinearIndex = rightShape.ComputeBroadcastIndex(indices);

            ref TScalar leftElement = ref left.ElementAt(leftLinearIndex);
            TScalar rightElement = right.ElementAt(rightLinearIndex);

            leftElement = operation(leftElement, rightElement);
        }
    }

    /// <summary>
    /// Represents a cumulative operation.
    /// </summary>
    public delegate TScalar CumulativeOperation<TScalar>(TScalar accumulator, TScalar element);

    /// <summary>
    /// Represents an unary operation.
    /// </summary>
    public delegate TScalar UnaryOperation<TScalar>(TScalar element);

    /// <summary>
    /// Represents a binary operation.
    /// </summary>
    public delegate TScalar BinaryOperation<TScalar>(TScalar left, TScalar right);
}
