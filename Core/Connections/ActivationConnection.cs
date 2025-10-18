using System.Numerics;
using NeuralNetworks.Core.Activations;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.Core.Connections;

public sealed class ActivationConnection<TScalar, TActivation> : IConnection<TScalar>
    where TScalar : struct, INumber<TScalar>
    where TActivation : IActivation<TScalar>
{
    public Tensor<TScalar> Input { get; }
    public Tensor<TScalar> Output { get; }

    public ActivationConnection(Tensor<TScalar> input, Tensor<TScalar> output)
    {
        if (input.Shape != output.Shape)
        {
            throw new IncompatibleShapeException(nameof(input), nameof(output));
        }

        Input = input;
        Output = output;
    }

    public void Calculate()
    {
        TensorShape shape = Input.Shape;
        int rank = shape.Rank;

        if (rank == 0)
        {
            Output[0] = TActivation.Activate(Input[0]);
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        Span<int> indices = stackalloc int[rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            Output[indices] = TActivation.Activate(Input[indices]);
        }
    }

    public Tensor<TScalar> Optimize(IOptimizer<TScalar> optimizer, in TensorSpan<TScalar> outputGradients)
    {
        TensorShape shape = Input.Shape;
        int rank = shape.Rank;

        Tensor<TScalar> inputGradients = Tensor.Create<TScalar>(shape);

        if (rank == 0)
        {
            inputGradients[0] = outputGradients[0] * TActivation.Derivate(Input[0]);
            return inputGradients;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        Span<int> indices = stackalloc int[rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            inputGradients[indices] = outputGradients[indices] * TActivation.Derivate(Input[indices]);
        }

        return inputGradients;
    }
}
