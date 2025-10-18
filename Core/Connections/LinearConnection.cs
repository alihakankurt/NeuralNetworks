using System.Numerics;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.Core.Connections;

public sealed class LinearConnection<TScalar> : IConnection<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    public Tensor<TScalar> Input { get; }
    public Tensor<TScalar> Output { get; }
    public Tensor<TScalar> Weights { get; }
    public Tensor<TScalar> Biases { get; }

    public LinearConnection(Tensor<TScalar> input, Tensor<TScalar> output)
    {
        TensorShape inputShape = input.Shape;
        TensorShape outputShape = output.Shape;

        if (inputShape.Rank != outputShape.Rank)
        {
            throw new IncompatibleShapeException(nameof(input), nameof(output));
        }

        int rank = inputShape.Rank;
        ReadOnlySpan<int> inputLengths = inputShape.Lengths;
        ReadOnlySpan<int> outputLengths = outputShape.Lengths;

        for (int dimension = rank - 2; dimension >= 0; --dimension)
        {
            if (inputLengths[dimension] != outputLengths[dimension])
            {
                throw new IncompatibleShapeException(nameof(input), nameof(output));
            }
        }

        scoped ReadOnlySpan<int> weightLengths = [];
        scoped ReadOnlySpan<int> biasLengths = [];

        if (rank > 0)
        {
            int inputFeatures = inputLengths[^1];
            int outputFeatures = outputLengths[^1];

            weightLengths = [outputFeatures, inputFeatures];
            biasLengths = [outputFeatures];
        }

        Input = input;
        Output = output;
        Weights = Tensor.Create<TScalar>(weightLengths);
        Biases = Tensor.Create<TScalar>(outputLengths);

        Weights.Randomize(-TScalar.One, TScalar.One);
        Biases.Randomize(-TScalar.One, TScalar.One);
    }

    public void Calculate()
    {
        TensorShape inputShape = Input.Shape;
        TensorShape outputShape = Output.Shape;
        int rank = inputShape.Rank;

        if (rank == 0)
        {
            Output[0] = Input[0] * Weights[0] + Biases[0];
            return;
        }

        int inputFeatures = inputShape.Lengths[^1];
        int outputFeatures = outputShape.Lengths[^1];

        if (rank == 1)
        {
            for (int outputIndex = 0; outputIndex < outputFeatures; ++outputIndex)
            {
                TScalar weightedSum = Biases[outputIndex];
                for (int inputIndex = 0; inputIndex < inputFeatures; ++inputIndex)
                {
                    TScalar inputValue = Input[inputIndex];

                    TScalar weight = Weights[outputIndex, inputIndex];
                    weightedSum += inputValue * weight;
                }

                Output[outputIndex] = weightedSum;
            }

            return;
        }

        ReadOnlySpan<int> lengths = inputShape.Lengths[..^1];

        Span<int> indices = stackalloc int[rank];
        Span<int> indicesToMove = indices[..^1];
        TensorShape.InitializeForwardIndexing(indicesToMove);

        while (TensorShape.MoveToNextIndex(lengths, indicesToMove))
        {
            for (int outputIndex = 0; outputIndex < outputFeatures; ++outputIndex)
            {
                TScalar weightedSum = Biases[outputIndex];
                for (int inputIndex = 0; inputIndex < inputFeatures; ++inputIndex)
                {
                    indices[^1] = inputIndex;
                    TScalar inputValue = Input[indices];

                    TScalar weight = Weights[outputIndex, inputIndex];
                    weightedSum += inputValue * weight;
                }

                indices[^1] = outputIndex;
                Output[indices] = weightedSum;
            }
        }
    }

    public Tensor<TScalar> Optimize(IOptimizer<TScalar> optimizer, in TensorSpan<TScalar> outputGradients)
    {
        TensorShape inputShape = Input.Shape;
        TensorShape outputShape = Output.Shape;
        int rank = inputShape.Rank;

        Tensor<TScalar> inputGradients = Tensor.Create<TScalar>(Input.Shape);
        Tensor<TScalar> weightGradients = Tensor.Create<TScalar>(Weights.Shape);
        Tensor<TScalar> biasGradients = Tensor.Create<TScalar>(Biases.Shape);

        if (rank == 0)
        {
            biasGradients[0] = outputGradients[0];
            weightGradients[0] = outputGradients[0] * Input[0];
            inputGradients[0] = outputGradients[0] * Weights[0];

            optimizer.Optimize(Weights, weightGradients.AsSpan());
            optimizer.Optimize(Biases, biasGradients.AsSpan());

            return inputGradients;
        }

        int inputFeatures = inputShape.Lengths[^1];
        int outputFeatures = outputShape.Lengths[^1];

        if (rank == 1)
        {
            for (int outputIndex = 0; outputIndex < outputFeatures; ++outputIndex)
            {
                TScalar outputValue = Output[outputIndex];
                TScalar outputGradient = outputGradients[outputIndex];
                biasGradients[outputIndex] += outputGradient;

                for (int inputIndex = 0; inputIndex < inputFeatures; ++inputIndex)
                {
                    TScalar inputValue = Input[inputIndex];
                    ReadOnlySpan<int> weightIndices = [outputIndex, inputIndex];

                    weightGradients[weightIndices] += outputGradient * inputValue;
                    inputGradients[inputIndex] += outputGradient * Weights[weightIndices];
                }
            }

            optimizer.Optimize(Weights, weightGradients.AsSpan());
            optimizer.Optimize(Biases, biasGradients.AsSpan());

            return inputGradients;
        }

        ReadOnlySpan<int> lengths = inputShape.Lengths[..^1];

        Span<int> indices = stackalloc int[rank];
        Span<int> indicesToMove = indices[..^1];
        TensorShape.InitializeForwardIndexing(indicesToMove);

        while (TensorShape.MoveToNextIndex(lengths, indicesToMove))
        {
            for (int outputIndex = 0; outputIndex < outputFeatures; ++outputIndex)
            {
                indices[^1] = outputIndex;
                TScalar outputValue = Output[indices];

                TScalar outputGradient = outputGradients[indices];
                biasGradients[outputIndex] += outputGradient;

                for (int inputIndex = 0; inputIndex < inputFeatures; ++inputIndex)
                {
                    indices[^1] = inputIndex;
                    TScalar inputValue = Input[indices];
                    ReadOnlySpan<int> weightIndices = [outputIndex, inputIndex];

                    weightGradients[weightIndices] += outputGradient * inputValue;
                    inputGradients[indices] += outputGradient * Weights[weightIndices];
                }
            }
        }

        optimizer.Optimize(Weights, weightGradients.AsSpan());
        optimizer.Optimize(Biases, biasGradients.AsSpan());

        return inputGradients;
    }
}
