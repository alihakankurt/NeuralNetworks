using System.Numerics;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.Core.Connections;

public abstract class ElementwiseConnectionBase<TScalar> : IConnection<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    public Tensor<TScalar> Input { get; }
    public Tensor<TScalar> Output { get; }

    public ElementwiseConnectionBase(Tensor<TScalar> input, Tensor<TScalar> output)
    {
        Guard.ShapesAreEqual(input.Shape, output.Shape);

        Input = input;
        Output = output;
    }

    public void Calculate()
    {
        TensorShape shape = Input.Shape;
        int rank = shape.Rank;

        if (rank == 0)
        {
            Output[0] = Calculate(Input[0]);
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        Span<int> indices = stackalloc int[rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            Output[indices] = Calculate(Input[indices]);
        }
    }

    public Tensor<TScalar> Optimize(IOptimizer<TScalar> optimizer, in TensorSpan<TScalar> outputGradients)
    {
        TensorShape shape = Input.Shape;
        int rank = shape.Rank;

        Tensor<TScalar> inputGradients = Tensor.Create<TScalar>(shape);

        if (rank == 0)
        {
            inputGradients[0] = outputGradients[0] * Derivate(Input[0]);
            return inputGradients;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        Span<int> indices = stackalloc int[rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            inputGradients[indices] = outputGradients[indices] * Derivate(Input[indices]);
        }

        return inputGradients;
    }

    protected abstract TScalar Calculate(TScalar value);
    protected abstract TScalar Derivate(TScalar value);
}
