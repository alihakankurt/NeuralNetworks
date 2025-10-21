using System.Runtime.CompilerServices;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core;

public sealed class IncompatibleShapeException : Exception
{
    public IncompatibleShapeException(string message) : base(message)
    {
    }
}

internal static class Guard
{
    internal static void ShapesAreEqual(TensorShape shape1, TensorShape shape2,
        [CallerArgumentExpression(nameof(shape1))] string? arg1 = null,
        [CallerArgumentExpression(nameof(shape2))] string? arg2 = null)
    {
        if (shape1.Rank != shape2.Rank)
        {
            throw new IncompatibleShapeException($"{arg1} and {arg2} are not equal");
        }
    }

    internal static void ShapesAreEqualExceptLastDimension(TensorShape shape1, TensorShape shape2,
        [CallerArgumentExpression(nameof(shape1))] string? arg1 = null,
        [CallerArgumentExpression(nameof(shape2))] string? arg2 = null)
    {
        if (shape1.Rank != shape2.Rank)
        {
            throw new IncompatibleShapeException($"{arg1} and {arg2} have different ranks");
        }

        int rank = shape1.Rank;
        ReadOnlySpan<int> inputLengths = shape1.Lengths;
        ReadOnlySpan<int> outputLengths = shape2.Lengths;

        for (int dimension = rank - 2; dimension >= 0; --dimension)
        {
            if (inputLengths[dimension] != outputLengths[dimension])
            {
                throw new IncompatibleShapeException($"{arg1} and {arg2} have a dimension with different length");
            }
        }
    }
}
