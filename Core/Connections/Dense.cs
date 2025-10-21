using System.Numerics;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.Core.Connections;

public sealed class Dense<TScalar> : IConnection<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    public Tensor<TScalar> Input { get; }
    public Tensor<TScalar> Output { get; }
    public Tensor<TScalar> Weights { get; }
    public Tensor<TScalar> Biases { get; }

    public Dense(Tensor<TScalar> input, Tensor<TScalar> output,
        TScalar minWeight = default, TScalar maxWeight = default,
        TScalar minBias = default, TScalar maxBias = default)
    {
        minWeight = (minWeight == default) ? -TScalar.One : minWeight;
        maxWeight = (maxWeight == default) ? TScalar.One : maxWeight;

        minBias = (minBias == default) ? -TScalar.One : minBias;
        maxBias = (maxBias == default) ? TScalar.One : maxBias;

        TensorShape inputShape = input.Shape;
        TensorShape outputShape = output.Shape;

        Guard.ShapesAreEqualExceptLastDimension(inputShape, outputShape);

        scoped ReadOnlySpan<int> weightLengths = [];
        scoped ReadOnlySpan<int> biasLengths = [];

        if (inputShape.Rank > 0)
        {
            int inputFeatures = inputShape.Lengths[^1];
            int outputFeatures = outputShape.Lengths[^1];

            weightLengths = [outputFeatures, inputFeatures];
            biasLengths = [outputFeatures];
        }

        Input = input;
        Output = output;

        Weights = Tensor.Create<TScalar>(weightLengths);
        Weights.Randomize(minWeight, maxWeight);

        Biases = Tensor.Create<TScalar>(biasLengths);
        Biases.Randomize(minBias, maxBias);
    }

    public void Calculate()
    {
        TensorShape inputShape = Input.Shape;
        TensorShape outputShape = Output.Shape;
        int rank = inputShape.Rank;

        if (rank == 0)
        {
            ref TScalar outputValue = ref Output[0];
            outputValue = Input[0] * Weights[0, 0] + Biases[0];
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
