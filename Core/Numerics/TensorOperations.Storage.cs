using System.Numerics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Numerics;

public static partial class Tensor
{
    /// <summary>
    /// Fills the <paramref name="tensor"/> with the <paramref name="value"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <param name="value">The value to fill.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Fill<TScalar>(this Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        tensor.AsSpan().Fill(value);
    }

    /// <summary>
    /// Fills the <paramref name="span"/> with the <paramref name="value"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="value">The value to fill.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Fill<TScalar>(this in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (_, v) => v;
        ExecuteIn<TScalar>(span, value, op);
    }

    /// <summary>
    /// Sets the values of <paramref name="destination"/> to the values of <paramref name="source"/>.
    /// </summary>
    /// <param name="destination">The destination tensor.</param>
    /// <param name="source">The source tensor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Set<TScalar>(this Tensor<TScalar> destination, Tensor<TScalar> source)
        where TScalar : struct, INumber<TScalar>
    {
        destination.AsSpan().Set(source.AsSpan());
    }

    /// <summary>
    /// Sets the values of <paramref name="destination"/> to the values of <paramref name="source"/>.
    /// </summary>
    /// <param name="destination">The destination tensor.</param>
    /// <param name="source">The source tensor span.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Set<TScalar>(this Tensor<TScalar> destination, TensorSpan<TScalar> source)
        where TScalar : struct, INumber<TScalar>
    {
        destination.AsSpan().Set(source);
    }

    /// <summary>
    /// Sets the values of <paramref name="destination"/> to the values of <paramref name="source"/>.
    /// </summary>
    /// <param name="destination">The destination tensor span.</param>
    /// <param name="source">The source tensor span.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Set<TScalar>(this in TensorSpan<TScalar> destination, in TensorSpan<TScalar> source)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (_, e2) => e2;
        ExecuteIn<TScalar>(destination, source, op);
    }

    /// <summary>
    /// Randomizes the values of <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor to randomize.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Randomize<TScalar>(this Tensor<TScalar> tensor, TScalar min, TScalar max)
        where TScalar : struct, INumber<TScalar>
    {
        Randomize(tensor.AsSpan(), min, max);
    }

    /// <summary>
    /// Randomizes the values of <paramref name="tensor"/>.
    /// </summary>
    /// <param name="span">The tensor span to randomize.</param>
    /// <param name="min">The minimum value.</param>
    /// <param name="max">The maximum value.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Randomize<TScalar>(this in TensorSpan<TScalar> span, TScalar min, TScalar max)
        where TScalar : struct, INumber<TScalar>
    {
        UnaryOperation<TScalar> op = (_) => TScalar.CreateTruncating(Random.Shared.NextDouble()) * (max - min) + min;
        ExecuteIn<TScalar>(span, op);
    }
}
