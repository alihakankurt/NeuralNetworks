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
        where TScalar : struct, INumber<TScalar>
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
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape shape = span.Shape;
        TScalar accumulator = initial;

        if (shape.Rank == 0)
        {
            TScalar element = span.ElementAt(0);

            accumulator = operation(accumulator, element);
            return accumulator;
        }

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

            TScalar element = span.ElementAt(linearIndex);

            accumulator = operation(accumulator, element);
        }

        return accumulator;
    }

    /// <summary>
    /// Executes an scalar operation over the <see cref="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="value">The scalar value.</param>
    /// <param name="operation">The scalar operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> Execute<TScalar>(this Tensor<TScalar> tensor, TScalar value, ScalarOperation<TScalar> operation)
        where TScalar : struct, INumber<TScalar>
    {
        return Tensor.Execute<TScalar>(tensor.AsSpan(), value, operation);
    }

    /// <summary>
    /// Executes an scalar operation over the <see cref="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="value">The scalar value.</param>
    /// <param name="operation">The scalar operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    public static Tensor<TScalar> Execute<TScalar>(this in TensorSpan<TScalar> span, TScalar value, ScalarOperation<TScalar> operation)
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape shape = span.Shape;
        var result = Tensor.Create<TScalar>(shape);

        if (shape.Rank == 0)
        {
            TScalar element = span.ElementAt(0);
            ref TScalar resultElement = ref result.ElementAt(0);

            resultElement = operation(element, value);
            return result;
        }

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

            TScalar element = span.ElementAt(linearIndex);
            ref TScalar resultElement = ref result.ElementAt(linearIndex);

            resultElement = operation(element, value);
        }

        return result;
    }

    /// <summary>
    /// Executes an scalar operation over the <see cref="tensor"/> in-place.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="value">The scalar value.</param>
    /// <param name="operation">The scalar operation to execute.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void ExecuteIn<TScalar>(this Tensor<TScalar> tensor, TScalar value, ScalarOperation<TScalar> operation)
        where TScalar : struct, INumber<TScalar>
    {
        Tensor.ExecuteIn(tensor.AsSpan(), value, operation);
    }

    /// <summary>
    /// Executes an scalar operation over the <see cref="span"/> in-place.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="value">The scalar value.</param>
    /// <param name="operation">The scalar operation to execute.</param>
    public static void ExecuteIn<TScalar>(this in TensorSpan<TScalar> span, TScalar value, ScalarOperation<TScalar> operation)
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape shape = span.Shape;

        if (shape.Rank == 0)
        {
            ref TScalar resultElement = ref span.ElementAt(0);

            resultElement = operation(resultElement, value);
            return;
        }

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            int linearIndex = shape.ComputeLinearIndexUnchecked(indices);

            ref TScalar resultElement = ref span.ElementAt(linearIndex);

            resultElement = operation(resultElement, value);
        }
    }

    /// <summary>
    /// Executes an unary operation over the <see cref="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="operation">The unary operation to execute.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> Execute<TScalar>(this Tensor<TScalar> tensor, UnaryOperation<TScalar> operation)
        where TScalar : struct, INumber<TScalar>
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
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape shape = span.Shape;
        var result = Tensor.Create<TScalar>(shape);

        if (shape.Rank == 0)
        {
            TScalar element = span.ElementAt(0);
            ref TScalar resultElement = ref result.ElementAt(0);

            resultElement = operation(element);
            return result;
        }

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
        where TScalar : struct, INumber<TScalar>
    {
        Tensor.ExecuteIn(tensor.AsSpan(), operation);
    }

    /// <summary>
    /// Executes an unary operation over the <see cref="span"/> in-place.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="operation">The unary operation to execute.</param>
    public static void ExecuteIn<TScalar>(this in TensorSpan<TScalar> span, UnaryOperation<TScalar> operation)
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape shape = span.Shape;

        if (shape.Rank == 0)
        {
            ref TScalar resultElement = ref span.ElementAt(0);

            resultElement = operation(resultElement);
            return;
        }

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
        where TScalar : struct, INumber<TScalar>
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
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape leftShape = left.Shape;
        TensorShape rightShape = right.Shape;

        if (!TensorShape.TryBroadcast(leftShape, rightShape, out var shape))
        {
            throw new InvalidOperationException($"The {nameof(left)} and {nameof(right)} has incompatible shapes to broadcast.");
        }

        var result = Tensor.Create<TScalar>(shape);

        if (shape.Rank == 0)
        {
            TScalar leftElement = left.ElementAt(0);
            TScalar rightElement = right.ElementAt(0);
            ref TScalar resultElement = ref result.ElementAt(0);

            resultElement = operation(leftElement, rightElement);
            return result;
        }

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
        where TScalar : struct, INumber<TScalar>
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
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape leftShape = left.Shape;
        TensorShape rightShape = right.Shape;

        if (!rightShape.IsBroadcastableTo(leftShape))
        {
            throw new InvalidOperationException($"The {nameof(left)} and {nameof(right)} has incompatible shapes to execute in-place operation.");
        }

        if (leftShape.Rank == 0)
        {
            ref TScalar leftElement = ref left.ElementAt(0);
            TScalar rightElement = right.ElementAt(0);

            leftElement = operation(leftElement, rightElement);
            return;
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
    /// Represents an operation that outputs a single value by accumulating the elemenets.
    /// </summary>
    public delegate TScalar CumulativeOperation<TScalar>(TScalar accumulator, TScalar element)
        where TScalar : struct, INumber<TScalar>;

    /// <summary>
    /// Represents an operation that executed for each element by a scalar value.
    /// </summary>
    public delegate TScalar ScalarOperation<TScalar>(TScalar element, TScalar value)
        where TScalar : struct, INumber<TScalar>;

    /// <summary>
    /// Represents an operation that executed for each element individually.
    /// </summary>
    public delegate TScalar UnaryOperation<TScalar>(TScalar element)
        where TScalar : struct, INumber<TScalar>;

    /// <summary>
    /// Represents an operation that executed for corresponding elements of left and right tensors.
    /// </summary>
    public delegate TScalar BinaryOperation<TScalar>(TScalar left, TScalar right)
        where TScalar : struct, INumber<TScalar>;
}
