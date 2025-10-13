using System.Numerics;

namespace NeuralNetworks.Core.Numerics;

/// <summary>
/// Provides common declarations to ensure consistency among implementations of multi-dimensional numerical data.
/// </summary>
/// <typeparam name="TScalar">The type of the numerical data.</typeparam>
public interface ITensor<TScalar>
    where TScalar : struct, INumberBase<TScalar>
{
    /// <summary>Gets the shape in the multi-dimensional space..</summary>
    TensorShape Shape { get; }

    /// <summary>Gets a value indicating whether there are no elements.</summary>
    bool IsEmpty { get; }

    /// <summary>Gets a value indicating whether there is exactly one element.</summary>
    bool IsScalar { get; }

    /// <summary>Gets a reference to the element where the <paramref name="indices"/> points to.</summary>
    ref TScalar this[params ReadOnlySpan<int> indices] { get; }

    /// <summary>Gets a new <see cref="TensorSpan{TScalar}"/> where the <paramref name="ranges"/> refers to.</summary>
    TensorSpan<TScalar> this[params ReadOnlySpan<Range> ranges] { get; }
}
