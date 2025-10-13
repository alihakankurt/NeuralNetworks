using System.Numerics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Numerics;

public static partial class Tensor
{
    /// <summary>
    /// Calculates sum of the elements of <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>The sum of the elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Sum<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumberBase<TScalar>
    {
        return Sum(tensor.AsSpan());
    }

    /// <summary>
    /// Calculates sum of the elements of <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <returns>The sum of the elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Sum<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumberBase<TScalar>
    {
        CumulativeOperation<TScalar> op = static (acc, e) => acc + e;
        return Accumulate<TScalar>(span, op, TScalar.AdditiveIdentity);
    }

    /// <summary>
    /// Calculates product of the elements of <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>The product of the elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Product<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumberBase<TScalar>
    {
        return Product(tensor.AsSpan());
    }

    /// <summary>
    /// Calculates product of the elements of <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <returns>The product of the elements.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Product<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumberBase<TScalar>
    {
        CumulativeOperation<TScalar> op = static (acc, e) => acc * e;
        return Accumulate<TScalar>(span, op, TScalar.MultiplicativeIdentity);
    }

    /// <summary>
    /// Negates the elements of <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Negate<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumberBase<TScalar>
    {
        Negate(tensor.AsSpan());
    }

    /// <summary>
    /// Negates the elements of <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Negate<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumberBase<TScalar>
    {
        UnaryOperation<TScalar> op = static (e) => -e;
        ExecuteIn<TScalar>(span, op);
    }

    /// <summary>
    /// Negates the elements of <paramref name="tensor"/> to a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> NegateTo<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumberBase<TScalar>
    {
        return NegateTo(tensor.AsSpan());
    }

    /// <summary>
    /// Negates the elements of <paramref name="span"/> to a new tensor.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> NegateTo<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumberBase<TScalar>
    {
        UnaryOperation<TScalar> op = static (e) => -e;
        return Execute<TScalar>(span, op);
    }

    /// <summary>
    /// Adds the elements of <paramref name="right"/> to <paramref name="left"/>.
    /// </summary>
    /// <param name="left">The tensor.</param>
    /// <param name="right">The tensor to add.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Add<TScalar>(this Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumberBase<TScalar>
    {
        Add(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Adds the elements of <paramref name="right"/> to <paramref name="left"/>.
    /// </summary>
    /// <param name="left">The tensor span.</param>
    /// <param name="right">The tensor span to add.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Add<TScalar>(this in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 + e2;
        ExecuteIn<TScalar>(left, right, op);
    }

    /// <summary>
    /// Adds the elements of <paramref name="left"/> and <paramref name="right"/> to a new tensor.
    /// </summary>
    /// <param name="left">The first tensor.</param>
    /// <param name="right">The second tensor.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> AddTo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumberBase<TScalar>
    {
        return AddTo(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Adds the elements of <paramref name="left"/> and <paramref name="right"/> to a new tensor.
    /// </summary>
    /// <param name="left">The first tensor span.</param>
    /// <param name="right">The second tensor span.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> AddTo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 + e2;
        return Execute<TScalar>(left, right, op);
    }

    /// <summary>
    /// Subtracts the elements of <paramref name="tensor2"/> from the elements of <paramref name="tensor1"/>.
    /// </summary>
    /// <param name="tensor1">The tensor to be subtracted.</param>
    /// <param name="tensor2">The tensor to subtract.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Subtract<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumberBase<TScalar>
    {
        Subtract(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Subtracts the elements of <paramref name="span2"/> from the elements of <paramref name="span1"/>.
    /// </summary>
    /// <param name="span1">The tensor span to be subtracted.</param>
    /// <param name="span2">The tensor span to subtract.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Subtract<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 - e2;
        ExecuteIn<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Subtracts the elements of <paramref name="tensor2"/> from the elements of <paramref name="tensor1"/> to a new tensor.
    /// </summary>
    /// <param name="tensor1">The tensor to be subtracted.</param>
    /// <param name="tensor2">The tensor to subtract.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> SubtractTo<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumberBase<TScalar>
    {
        return SubtractTo(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Subtracts the elements of <paramref name="span2"/> from the elements of <paramref name="span1"/> to a new tensor.
    /// </summary>
    /// <param name="span1">The tensor span to be subtracted.</param>
    /// <param name="span2">The tensor span to subtract.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> SubtractTo<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 - e2;
        return Execute<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Multiplies the elements of <paramref name="tensor1"/> and <paramref name="tensor2"/>.
    /// </summary>
    /// <param name="tensor1">The first tensor.</param>
    /// <param name="tensor2">The second tensor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumberBase<TScalar>
    {
        Multiply(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Multiplies the elements of <paramref name="span1"/> and <paramref name="span2"/>.
    /// </summary>
    /// <param name="span1">The first tensor span.</param>
    /// <param name="span2">The second tensor span.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 * e2;
        ExecuteIn<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Multiplies the elements of <paramref name="tensor1"/> and <paramref name="tensor2"/> to a new tensor.
    /// </summary>
    /// <param name="tensor1">The first tensor.</param>
    /// <param name="tensor2">The second tensor.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> MultiplyTo<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumberBase<TScalar>
    {
        return MultiplyTo(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Multiplies the elements of <paramref name="span1"/> and <paramref name="span2"/> to a new tensor.
    /// </summary>
    /// <param name="span1">The first tensor span.</param>
    /// <param name="span2">The second tensor span.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> MultiplyTo<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 * e2;
        return Execute<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Divides the values of <paramref name="tensor1"/> by the values of <paramref name="tensor2"/>.
    /// </summary>
    /// <param name="tensor1">The tensor to be divided.</param>
    /// <param name="tensor2">The tensor to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Divide<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumberBase<TScalar>
    {
        Divide(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Divides the values of <paramref name="span1"/> by the values of <paramref name="span2"/>.
    /// </summary>
    /// <param name="span1">The tensor span to be divided.</param>
    /// <param name="span2">The tensor span to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Divide<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 / e2;
        ExecuteIn<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Divides the values of <paramref name="tensor1"/> by the values of <paramref name="tensor2"/> to a new tensor.
    /// </summary>
    /// <param name="tensor1">The tensor to be divided.</param>
    /// <param name="tensor2">The tensor to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> DivideTo<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumberBase<TScalar>
    {
        return DivideTo(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Divides the values of <paramref name="span1"/> by the values of <paramref name="span2"/> to a new tensor.
    /// </summary>
    /// <param name="span1">The tensor span to be divided.</param>
    /// <param name="span2">The tensor span to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> DivideTo<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumberBase<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 / e2;
        return Execute<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Divides the values of <paramref name="tensor1"/> by the values of <paramref name="tensor2"/> and computes the remainders.
    /// </summary>
    /// <param name="tensor1">The tensor to be divided.</param>
    /// <param name="tensor2">The tensor to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Modulo<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumber<TScalar>
    {
        Modulo(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Divides the values of <paramref name="span1"/> by the values of <paramref name="span2"/> and computes the remainders.
    /// </summary>
    /// <param name="span1">The tensor span to be divided.</param>
    /// <param name="span2">The tensor span to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Modulo<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 % e2;
        ExecuteIn<TScalar>(span1, span2, op);
    }

    /// <summary>
    /// Divides the values of <paramref name="tensor1"/> by the values of <paramref name="tensor2"/> and computes the remainders to a new tensor.
    /// </summary>
    /// <param name="tensor1">The tensor to be divided.</param>
    /// <param name="tensor2">The tensor to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> ModuloTo<TScalar>(Tensor<TScalar> tensor1, Tensor<TScalar> tensor2)
        where TScalar : struct, INumber<TScalar>
    {
        return ModuloTo(tensor1.AsSpan(), tensor2.AsSpan());
    }

    /// <summary>
    /// Divides the values of <paramref name="span1"/> by the values of <paramref name="span2"/> and computes the remainders to a new tensor.
    /// </summary>
    /// <param name="span1">The tensor span to be divided.</param>
    /// <param name="span2">The tensor span to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> ModuloTo<TScalar>(in TensorSpan<TScalar> span1, in TensorSpan<TScalar> span2)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (e1, e2) => e1 % e2;
        return Execute<TScalar>(span1, span2, op);
    }
}
