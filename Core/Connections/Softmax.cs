using System.Numerics;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.Core.Connections;

public sealed class Softmax<TScalar> : IConnection<TScalar>
    where TScalar : struct, IFloatingPointIeee754<TScalar>
{
    public Tensor<TScalar> Input { get; }

    public Tensor<TScalar> Output { get; }

    public Softmax(Tensor<TScalar> input, Tensor<TScalar> output)
    {
        Guard.ShapesAreEqualExceptLastDimension(input.Shape, output.Shape);

        Input = input;
        Output = output;
    }

    public void Calculate()
    {
        TensorShape shape = Input.Shape;
        int rank = shape.Rank;

        if (rank == 0)
        {
            Output[0] = TScalar.One;
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;
        int features = lengths[^1];

        if (rank == 1)
        {
            Calculate([0], features);
            return;
        }

        lengths = lengths[..^1];
        Span<int> indices = stackalloc int[rank];

        Span<int> indicesToMove = indices[..^1];
        TensorShape.InitializeForwardIndexing(indicesToMove);

        while (TensorShape.MoveToNextIndex(lengths, indicesToMove))
        {
            Calculate(indices, features);
        }
    }

    private void Calculate(Span<int> indices, int features)
    {
        TScalar maxValue = TScalar.NegativeInfinity;
        for (int index = 0; index < features; ++index)
        {
            indices[^1] = index;
            maxValue = TScalar.Max(maxValue, Input[indices]);
        }

        TScalar outputSum = TScalar.Zero;
        for (int index = 0; index < features; ++index)
        {
            indices[^1] = index;
            TScalar inputValue = Input[indices];
            ref TScalar outputValue = ref Output[indices];

            outputValue = TScalar.Exp(inputValue - maxValue);
            outputSum += outputValue;
        }

        for (int index = 0; index < features; ++index)
        {
            indices[^1] = index;
            ref TScalar outputValue = ref Output[indices];
            outputValue /= outputSum;
        }
    }

    public Tensor<TScalar> Optimize(IOptimizer<TScalar> optimizer, in TensorSpan<TScalar> outputGradients)
    {
        TensorShape shape = Input.Shape;
        int rank = shape.Rank;

        var inputGradients = Tensor.Create<TScalar>(Input.Shape);
        if (rank == 0)
        {
            return inputGradients;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;
        int features = lengths[^1];

        if (rank == 1)
        {
            Optimize([0], features, outputGradients, inputGradients.AsSpan());
            return inputGradients;
        }

        lengths = lengths[..^1];
        Span<int> indices = stackalloc int[rank];

        Span<int> indicesToMove = indices[..^1];
        TensorShape.InitializeForwardIndexing(indicesToMove);

        while (TensorShape.MoveToNextIndex(lengths, indicesToMove))
        {
            Optimize(indices, features, outputGradients, inputGradients.AsSpan());
        }

        return inputGradients;
    }

    private void Optimize(Span<int> indices, int features, TensorSpan<TScalar> outputGradients, TensorSpan<TScalar> inputGradients)
    {
        TScalar dot = TScalar.Zero;
        for (int index = 0; index < features; ++index)
        {
            indices[^1] = index;
            dot += Output[indices] * outputGradients[indices];
        }

        for (int index = 0; index < features; ++index)
        {
            indices[^1] = index;
            inputGradients[indices] = Output[indices] * (outputGradients[indices] - dot);
        }
    }
}
