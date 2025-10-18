using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Losses;

public readonly struct MeanSquaredLoss<TScalar> : ILoss<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    public static TScalar Calculate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs)
    {
        TensorShape shape = outputs.Shape;
        if (shape != expectedOutputs.Shape)
        {
            throw new IncompatibleShapeException(nameof(outputs), nameof(expectedOutputs));
        }

        if (shape.Rank == 0)
        {
            TScalar loss = outputs[0] - expectedOutputs[0];
            return loss * loss;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        TScalar squaredLossSum = TScalar.Zero;

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            TScalar loss = outputs[indices] - expectedOutputs[indices];
            squaredLossSum += loss * loss;
        }

        return squaredLossSum / TScalar.CreateTruncating(shape.ElementCount);
    }

    public static void Gradient(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs, in TensorSpan<TScalar> gradients)
    {
        TensorShape shape = outputs.Shape;
        if (shape != expectedOutputs.Shape)
        {
            throw new IncompatibleShapeException(nameof(outputs), nameof(expectedOutputs));
        }

        TScalar two = TScalar.CreateTruncating(2);
        if (shape.Rank == 0)
        {
            gradients[0] += two * (outputs[0] - expectedOutputs[0]);
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;
        TScalar elementCount = TScalar.CreateTruncating(shape.ElementCount);

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            TScalar outputValue = outputs[indices];
            TScalar expectedValue = expectedOutputs[indices];

            gradients[indices] += two * (outputValue - expectedValue) / elementCount;
        }
    }
}
