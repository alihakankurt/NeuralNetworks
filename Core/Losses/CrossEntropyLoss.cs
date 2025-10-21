using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Losses;

public readonly struct CrossEntropyLoss<TScalar> : ILoss<TScalar>
    where TScalar : struct, IFloatingPointIeee754<TScalar>
{
    public static TScalar Calculate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs)
    {
        Guard.ShapesAreEqual(outputs.Shape, expectedOutputs.Shape);

        TensorShape shape = outputs.Shape;
        if (shape.Rank == 0)
        {
            TScalar output = outputs[0];
            TScalar expected = expectedOutputs[0];

            output = TScalar.Clamp(output, TScalar.Epsilon, TScalar.One - TScalar.Epsilon);
            TScalar outputComplement = TScalar.One - output;
            TScalar expectedComplement = TScalar.One - expected;

            TScalar loss = expected * TScalar.Log(output) - expectedComplement * TScalar.Log(outputComplement);
            return -loss;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;
        TScalar lossSum = TScalar.Zero;

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            TScalar output = outputs[indices];
            TScalar expected = expectedOutputs[indices];

            output = TScalar.Max(output, TScalar.Epsilon);

            lossSum += expected * TScalar.Log(output);
        }

        TScalar averageLoss = lossSum / TScalar.CreateTruncating(shape.ElementCount);
        return -averageLoss;
    }

    public static Tensor<TScalar> Derivate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs)
    {
        var gradients = Tensor.Create<TScalar>(outputs.Shape);
        Derivate(outputs, expectedOutputs, gradients.AsSpan());
        return gradients;
    }

    public static void Derivate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs, in TensorSpan<TScalar> gradients)
    {
        Guard.ShapesAreEqual(outputs.Shape, expectedOutputs.Shape);

        TensorShape shape = outputs.Shape;

        if (shape.Rank == 0)
        {
            TScalar output = outputs[0];
            TScalar expected = expectedOutputs[0];

            gradients[0] += output - expected;
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        int elementCount = shape.ElementCount / lengths[^1];
        TScalar elementFactor = TScalar.One / TScalar.CreateTruncating(elementCount);

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            TScalar output = outputs[indices];
            TScalar expected = expectedOutputs[indices];

            gradients[indices] += (output - expected) * elementFactor;
        }
    }
}
