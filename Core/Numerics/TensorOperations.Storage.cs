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
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Fill<TScalar>(this Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumberBase<TScalar>
    {
        tensor.AsSpan().Fill(value);
    }

    /// <summary>
    /// Fills the <paramref name="span"/> with the <paramref name="value"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <param name="value">The value to fill.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Fill<TScalar>(this in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumberBase<TScalar>
    {
        UnaryOperation<TScalar> op = (_) => value;
        ExecuteIn<TScalar>(span, op);
    }

    /// <summary>
    /// Sets the values of <paramref name="destination"/> to the values of <paramref name="source"/>.
    /// </summary>
    /// <param name="destination">The destination tensor.</param>
    /// <param name="source">The source tensor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Set<TScalar>(this Tensor<TScalar> destination, Tensor<TScalar> source)
        where TScalar : struct, INumberBase<TScalar>
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
        where TScalar : struct, INumberBase<TScalar>
    {
        destination.AsSpan().Set(source);
    }

    /// <summary>
    /// Sets the values of <paramref name="destination"/> to the values of <paramref name="source"/>.
    /// </summary>
    /// <param name="destination">The destination tensor span.</param>
    /// <param name="source">The source tensor span.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Set<TScalar>(this in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (_, e2) => e2;
        ExecuteIn<TScalar>(span1, span2, op);
    }
}
