using System.Numerics;
using System.Runtime.CompilerServices;

namespace NeuralNetworks.Core.Numerics;

public static partial class Tensor
{
    /// <summary>
    /// Calculates the sum of the elements of the <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>A single <typeparamref name="TScalar"/> representing the sum.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Sum<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumber<TScalar>
    {
        return Sum(tensor.AsSpan());
    }

    /// <summary>
    /// Calculates the sum of the elements of the <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <returns>A single <typeparamref name="TScalar"/> representing the sum.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Sum<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumber<TScalar>
    {
        CumulativeOperation<TScalar> op = static (acc, e) => acc + e;
        return Accumulate<TScalar>(span, op, TScalar.AdditiveIdentity);
    }

    /// <summary>
    /// Calculates the product of the elements of the <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>A single <typeparamref name="TScalar"/> representing the product.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Product<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumber<TScalar>
    {
        return Product(tensor.AsSpan());
    }

    /// <summary>
    /// Calculates the product of the elements of the <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <returns>A single <typeparamref name="TScalar"/> representing the product.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static TScalar Product<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumber<TScalar>
    {
        CumulativeOperation<TScalar> op = static (acc, e) => acc * e;
        return Accumulate<TScalar>(span, op, TScalar.MultiplicativeIdentity);
    }

    /// <summary>
    /// Negates the elements of the <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Negate<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumber<TScalar>
    {
        Negate(tensor.AsSpan());
    }

    /// <summary>
    /// Negates the elements of the <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Negate<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumber<TScalar>
    {
        UnaryOperation<TScalar> op = static (e) => -e;
        ExecuteIn<TScalar>(span, op);
    }

    /// <summary>
    /// Negates the elements of the <paramref name="tensor"/> into a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> NegateTo<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumber<TScalar>
    {
        return NegateTo(tensor.AsSpan());
    }

    /// <summary>
    /// Negates the elements of the <paramref name="span"/> into a new tensor.
    /// </summary>
    /// <param name="span">The tensor span.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> NegateTo<TScalar>(this in TensorSpan<TScalar> span)
        where TScalar : struct, INumber<TScalar>
    {
        UnaryOperation<TScalar> op = static (e) => -e;
        return Execute<TScalar>(span, op);
    }

    /// <summary>
    /// Adds the elements of the <paramref name="right"/> to the <paramref name="left"/>.
    /// </summary>
    /// <param name="left">The tensor to be added.</param>
    /// <param name="right">The tensor to add.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Add<TScalar>(this Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        Add(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Adds the elements of the <paramref name="right"/> to the <paramref name="left"/>.
    /// </summary>
    /// <param name="left">The tensor span to be added.</param>
    /// <param name="right">The tensor span to add.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Add<TScalar>(this in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l + r;
        ExecuteIn<TScalar>(left, right, op);
    }

    /// <summary>
    /// Adds the <paramref name="value"/> to the elements of the <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor to be added.</param>
    /// <param name="value">The value to add.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Add<TScalar>(this Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        Add(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Adds the <paramref name="value"/> to the elements of the <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span to be added.</param>
    /// <param name="value">The value to add.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Add<TScalar>(this in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e + v;
        ExecuteIn<TScalar>(span, value, op);
    }

    /// <summary>
    /// Adds the elements of the <paramref name="left"/> and the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The first tensor to add.</param>
    /// <param name="right">The second tensor to add.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> AddTo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        return AddTo(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Adds the elements of the <paramref name="left"/> and the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The first tensor span to add.</param>
    /// <param name="right">The second tensor span to add.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> AddTo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l + r;
        return Execute<TScalar>(left, right, op);
    }

    /// <summary>
    /// Adds the <paramref name="value"/> to the elements of the <paramref name="tensor"/> into a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor to add.</param>
    /// <param name="value">The value to add.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> AddTo<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        return AddTo(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Adds the <paramref name="value"/> to the elements of the <paramref name="span"/> into a new tensor.
    /// </summary>
    /// <param name="span">The tensor span to add.</param>
    /// <param name="value">The value to add.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> AddTo<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e + v;
        return Execute<TScalar>(span, value, op);
    }

    /// <summary>
    /// Subtracts the elements of the <paramref name="right"/> from the <paramref name="left"/>.
    /// </summary>
    /// <param name="left">The tensor to be subtracted.</param>
    /// <param name="right">The tensor to subtract.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Subtract<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        Subtract(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Subtracts the elements of the <paramref name="right"/> from the <paramref name="left"/>.
    /// </summary>
    /// <param name="left">The tensor span to be subtracted.</param>
    /// <param name="right">The tensor span to subtract.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Subtract<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l - r;
        ExecuteIn<TScalar>(left, right, op);
    }

    /// <summary>
    /// Subtracts the <paramref name="value"/> from the elements of the <paramref name="tensor"/>.
    /// </summary>
    /// <param name="tensor">The tensor to be subtracted.</param>
    /// <param name="value">The value to subtract.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Subtract<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        Subtract(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Subtracts the <paramref name="value"/> from the elements of the <paramref name="span"/>.
    /// </summary>
    /// <param name="span">The tensor span to be subtracted.</param>
    /// <param name="value">The value to subtract.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Subtract<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e - v;
        ExecuteIn<TScalar>(span, value, op);
    }

    /// <summary>
    /// Subtracts the elements of the <paramref name="right"/> from the <paramref name="left"/> into a new tensor.
    /// </summary>
    /// <param name="left">The tensor to be subtracted.</param>
    /// <param name="right">The tensor to subtract.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> SubtractTo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        return SubtractTo(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Subtracts the elements of the <paramref name="right"/> from the <paramref name="left"/> into a new tensor.
    /// </summary>
    /// <param name="left">The tensor span to be subtracted.</param>
    /// <param name="right">The tensor span to subtract.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> SubtractTo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l - r;
        return Execute<TScalar>(left, right, op);
    }

    /// <summary>
    /// Subtracts the <paramref name="value"/> from the elements of the <paramref name="tensor"/> into a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor to be subtracted.</param>
    /// <param name="value">The value to subtract.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> SubtractTo<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        return SubtractTo(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Subtracts the <paramref name="value"/> from the elements of the <paramref name="span"/> into a new tensor.
    /// </summary>
    /// <param name="span">The tensor span to be subtracted.</param>
    /// <param name="value">The value to subtract.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> SubtractTo<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e - v;
        return Execute<TScalar>(span, value, op);
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="left"/> by the <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The tensor to be multiplied.</param>
    /// <param name="right">The tensor to multiply.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        Multiply(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="left"/> by the <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The tensor span to be multiplied.</param>
    /// <param name="right">The tensor span to multiply.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l * r;
        ExecuteIn<TScalar>(left, right, op);
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="tensor"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="tensor">The tensor to be multiplied.</param>
    /// <param name="value">The value to multiply.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        Multiply(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="span"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="span">The tensor span to be multiplied.</param>
    /// <param name="value">The value to multiply.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Multiply<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e * v;
        ExecuteIn<TScalar>(span, value, op);
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="left"/> and the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The first tensor to multiply.</param>
    /// <param name="right">The second tensor to multiply.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> MultiplyTo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        return MultiplyTo(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="left"/> and the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The first tensor span to multiply.</param>
    /// <param name="right">The second tensor span to multiply.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> MultiplyTo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l * r;
        return Execute<TScalar>(left, right, op);
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="tensor"/> by the <paramref name="value"/> into a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor to multiply.</param>
    /// <param name="value">The value to multiply.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> MultiplyTo<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        return MultiplyTo(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Multiplies the elements of the <paramref name="span"/> by the <paramref name="value"/> into a new tensor.
    /// </summary>
    /// <param name="span">The tensor span to multiply.</param>
    /// <param name="value">The value to multiply.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> MultiplyTo<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e * v;
        return Execute<TScalar>(span, value, op);
    }

    /// <summary>
    /// Divides the elements of the <paramref name="left"/> by the <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The tensor to be divided.</param>
    /// <param name="right">The tensor to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Divide<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        Divide(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Divides the elements of the <paramref name="left"/> by the <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The tensor span to be divided.</param>
    /// <param name="right">The tensor span to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Divide<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l / r;
        ExecuteIn<TScalar>(left, right, op);
    }

    /// <summary>
    /// Divides the elements of the <paramref name="tensor"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="tensor">The tensor to be divided.</param>
    /// <param name="value">The value to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Divide<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        Divide(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Divides the elements of the <paramref name="span"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="span">The tensor span to be divided.</param>
    /// <param name="value">The value to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Divide<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e / v;
        ExecuteIn<TScalar>(span, value, op);
    }

    /// <summary>
    /// Divides the elements of the <paramref name="left"/> by the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The tensor to be divided.</param>
    /// <param name="right">The tensor to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> DivideTo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        return DivideTo(left.AsSpan(), right.AsSpan());
    }


    /// <summary>
    /// Divides the elements of the <paramref name="left"/> by the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The tensor span to be divided.</param>
    /// <param name="right">The tensor span to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> DivideTo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l / r;
        return Execute<TScalar>(left, right, op);
    }

    /// <summary>
    /// Divides the elements of the <paramref name="tensor"/> by the <paramref name="value"/> into a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor to be divided.</param>
    /// <param name="value">The value to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> DivideTo<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        return DivideTo(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Divides the elements of the <paramref name="span"/> by the <paramref name="value"/> into a new tensor.
    /// </summary>
    /// <param name="span">The tensor span to be divided.</param>
    /// <param name="value">The value to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> DivideTo<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e / v;
        return Execute<TScalar>(span, value, op);
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="left"/> by the <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The tensor to be divided.</param>
    /// <param name="right">The tensor to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Modulo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        Modulo(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="left"/> by the <paramref name="right"/>.
    /// </summary>
    /// <param name="left">The tensor span to be divided.</param>
    /// <param name="right">The tensor span to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Modulo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l % r;
        ExecuteIn<TScalar>(left, right, op);
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="tensor"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="tensor">The tensor to be divided.</param>
    /// <param name="value">The value to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Modulo<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        Modulo(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="span"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="span">The tensor span to be divided.</param>
    /// <param name="value">The value to divide.</param>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static void Modulo<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e % v;
        ExecuteIn<TScalar>(span, value, op);
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="left"/> by the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The tensor to be divided.</param>
    /// <param name="right">The tensor to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> ModuloTo<TScalar>(Tensor<TScalar> left, Tensor<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        return ModuloTo(left.AsSpan(), right.AsSpan());
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="left"/> by the <paramref name="right"/> into a new tensor.
    /// </summary>
    /// <param name="left">The tensor span to be divided.</param>
    /// <param name="right">The tensor span to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> ModuloTo<TScalar>(in TensorSpan<TScalar> left, in TensorSpan<TScalar> right)
        where TScalar : struct, INumber<TScalar>
    {
        BinaryOperation<TScalar> op = static (l, r) => l % r;
        return Execute<TScalar>(left, right, op);
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="tensor"/> by the <paramref name="value"/> into a new tensor.
    /// </summary>
    /// <param name="tensor">The tensor to be divided.</param>
    /// <param name="value">The value to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> ModuloTo<TScalar>(Tensor<TScalar> tensor, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        return ModuloTo(tensor.AsSpan(), value);
    }

    /// <summary>
    /// Takes the modulo of the elements of the <paramref name="span"/> by the <paramref name="value"/>.
    /// </summary>
    /// <param name="span">The tensor span to be divided.</param>
    /// <param name="value">The value to divide.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining | MethodImplOptions.AggressiveOptimization)]
    public static Tensor<TScalar> ModuloTo<TScalar>(in TensorSpan<TScalar> span, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        ScalarOperation<TScalar> op = static (e, v) => e % v;
        return Execute<TScalar>(span, value, op);
    }
}
