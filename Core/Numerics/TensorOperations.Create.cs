using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace NeuralNetworks.Core.Numerics;

public static partial class Tensor
{
    /// <summary>
    /// Creates a new <see cref="Tensor{TScalar}"/> with the specified <paramref name="shape"/>.
    /// </summary>
    /// <param name="shape">The multi-dimensional shape.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    public static Tensor<TScalar> Create<TScalar>(TensorShape shape)
        where TScalar : struct, INumber<TScalar>
    {
        TScalar[] storage = [];
        int elementCount = shape.ElementCount;

        if (elementCount > 0)
        {
            storage = GC.AllocateArray<TScalar>(elementCount);
        }

        return new Tensor<TScalar>(shape, storage);
    }

    /// <summary>
    /// Creates a new <see cref="Tensor{TScalar}"/> with the specified <paramref name="shape"/>
    /// and initializes the storage with the <see cref="value"/>.
    /// </summary>
    /// <param name="shape">The multi-dimensional shape.</param>
    /// <param name="value">The value to initialize storage.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    public static Tensor<TScalar> Create<TScalar>(TensorShape shape, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        TScalar[] storage = [];
        int elementCount = shape.ElementCount;

        if (elementCount > 0)
        {
            storage = GC.AllocateUninitializedArray<TScalar>(elementCount);
            storage.AsSpan().Fill(value);
        }

        return new Tensor<TScalar>(shape, storage);
    }

    /// <summary>
    /// Creates a new <see cref="Tensor{TScalar}"/> with a shape created by the specified <paramref name="lengths"/>.
    /// </summary>
    /// <param name="lengths">The dimensional lengths of the shape.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> Create<TScalar>(ReadOnlySpan<int> lengths)
        where TScalar : struct, INumber<TScalar>
    {
        var shape = TensorShape.Create(lengths);
        return Tensor.Create<TScalar>(shape);
    }

    /// <summary>
    /// Creates a new <see cref="Tensor{TScalar}"/> with a shape created by the specified <paramref name="lengths"/>
    /// and initializes the storage with the <paramref name="value"/>.
    /// </summary>
    /// <param name="lengths">The dimensional lengths of the shape.</param>
    /// <param name="value">The value to initialize storage.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> Create<TScalar>(ReadOnlySpan<int> lengths, TScalar value)
        where TScalar : struct, INumber<TScalar>
    {
        var shape = TensorShape.Create(lengths);
        return Tensor.Create<TScalar>(shape, value);
    }

    /// <summary>
    /// Creates a new 1-dimensional <see cref="Tensor{TScalar}"/> with the specified values.
    /// </summary>
    /// <param name="source">The span of values.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> CreateFrom<TScalar>(params ReadOnlySpan<TScalar> source)
        where TScalar : struct, INumber<TScalar>
    {
        TScalar[] storage = [];
        TensorShape shape = TensorShape.Empty;

        if (source.Length > 0)
        {
            shape = TensorShape.Create([source.Length]);
            storage = GC.AllocateUninitializedArray<TScalar>(shape.ElementCount);
            source.CopyTo(storage.AsSpan());
        }

        return new Tensor<TScalar>(shape, storage);
    }

    /// <summary>
    /// Creates a new 1-dimensional <see cref="Tensor{TScalar}"/> with the specified values.
    /// </summary>
    /// <param name="source">The list of values.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> CreateFrom<TScalar>(List<TScalar> source)
        where TScalar : struct, INumber<TScalar>
    {
        return Tensor.CreateFrom<TScalar>(CollectionsMarshal.AsSpan(source));
    }

    /// <summary>
    /// Creates a new 2-dimensional <see cref="Tensor{TScalar}"/> with the specified values.
    /// </summary>
    /// <param name="source">The span of value collections.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> CreateFrom<TScalar>(params ReadOnlySpan<TScalar[]> source)
        where TScalar : struct, INumber<TScalar>
    {
        TScalar[] storage = [];
        TensorShape shape = TensorShape.Empty;

        if (source.Length > 0)
        {
            shape = TensorShape.Create([source.Length, source[0].Length]);
            storage = GC.AllocateUninitializedArray<TScalar>(shape.ElementCount);

            for (int y = 0; y < source.Length; ++y)
            {
                ArgumentOutOfRangeException.ThrowIfNotEqual(source[y].Length, source[0].Length);
                source[y].CopyTo(storage.AsSpan(y * source[0].Length, source[0].Length));
            }
        }

        return new Tensor<TScalar>(shape, storage);
    }

    /// <summary>
    /// Creates a new 2-dimensional <see cref="Tensor{TScalar}"/> with the specified values.
    /// </summary>
    /// <param name="source">The list of value collections.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/>.</returns>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Tensor<TScalar> CreateFrom<TScalar>(List<TScalar[]> source)
        where TScalar : struct, INumber<TScalar>
    {
        return Tensor.CreateFrom<TScalar>(CollectionsMarshal.AsSpan(source));
    }

    /// <summary>
    /// Creates a clone of the tensor instance.
    /// </summary>
    /// <param name="tensor">The tensor to clone.</param>
    /// <returns>A new instance of <see cref="Tensor{TScalar}"/> as clone</returns>
    public static Tensor<TScalar> Clone<TScalar>(this Tensor<TScalar> tensor)
        where TScalar : struct, INumber<TScalar>
    {
        TensorShape shape = tensor.Shape;
        TScalar[] storage = [];

        if (shape.ElementCount > 0)
        {
            storage = GC.AllocateUninitializedArray<TScalar>(shape.ElementCount);
            MemoryMarshal.CreateReadOnlySpan(ref tensor.ElementAt(0), shape.ElementCount)
                .CopyTo(storage.AsSpan());
        }

        return new Tensor<TScalar>(shape, storage);
    }
}
