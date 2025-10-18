using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Optimizers;

public sealed class MomentumSGD<TScalar> : IOptimizer<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    private readonly Dictionary<Tensor<TScalar>, Tensor<TScalar>> _velocities;
    private readonly TScalar _learningRate;
    private readonly TScalar _momentum;

    public MomentumSGD(TScalar learningRate, TScalar momentum)
    {
        _velocities = [];
        _learningRate = learningRate;
        _momentum = momentum;
    }

    public void Optimize(Tensor<TScalar> parameters, in TensorSpan<TScalar> gradients)
    {
        TensorShape shape = parameters.Shape;
        if (shape != gradients.Shape)
        {
            throw new IncompatibleShapeException(nameof(parameters), nameof(gradients));
        }

        if (!_velocities.TryGetValue(parameters, out Tensor<TScalar>? velocity))
        {
            velocity = Tensor.Create<TScalar>(parameters.Shape);
            _velocities.Add(parameters, velocity);
        }

        if (shape.Rank == 0)
        {
            ref TScalar v = ref velocity[0];
            v = _momentum * v - _learningRate * gradients[0];
            parameters[0] += v;
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;

        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            ref TScalar v = ref velocity[indices];
            v = _momentum * v - _learningRate * gradients[indices];
            parameters[indices] += v;
        }
    }

    public void Clean()
    {
        _velocities.Clear();
    }
}
