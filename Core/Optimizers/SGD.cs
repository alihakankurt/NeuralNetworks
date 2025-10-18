using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Optimizers;

public sealed class SGD<TScalar> : IOptimizer<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    private readonly TScalar _learningRate;

    public SGD(TScalar learningRate)
    {
        _learningRate = learningRate;
    }

    public void Dispose()
    {
    }

    public void Optimize(Tensor<TScalar> parameters, in TensorSpan<TScalar> gradients)
    {
        TensorShape shape = parameters.Shape;
        if (shape != gradients.Shape)
        {
            throw new IncompatibleShapeException(nameof(parameters), nameof(gradients));
        }

        if (shape.Rank == 0)
        {
            parameters[0] -= _learningRate * gradients[0];
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (shape.MoveToNextIndex(indices))
        {
            parameters[indices] -= _learningRate * gradients[indices];
        }
    }
}
