using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Optimizers;

public sealed class Adam<TScalar> : IOptimizer<TScalar>
    where TScalar : struct, IFloatingPointIeee754<TScalar>
{
    private readonly Dictionary<Tensor<TScalar>, AdamState> _states;
    private readonly TScalar _learningRate;
    private readonly TScalar _beta1;
    private readonly TScalar _beta2;

    public Adam(TScalar learningRate, TScalar beta1 = default, TScalar beta2 = default)
    {
        _states = [];
        _learningRate = learningRate;
        _beta1 = (beta1 == default) ? TScalar.CreateTruncating(0.9f) : beta1;
        _beta2 = (beta2 == default) ? TScalar.CreateTruncating(0.999f) : beta2;
    }

    public void Optimize(Tensor<TScalar> parameters, in TensorSpan<TScalar> gradients)
    {
        Guard.ShapesAreEqual(parameters.Shape, gradients.Shape);

        TensorShape shape = parameters.Shape;

        if (!_states.TryGetValue(parameters, out AdamState state))
        {
            state = new AdamState(parameters.Shape);
            _states.Add(parameters, state);
        }

        TScalar time = TScalar.CreateTruncating(++state.Time);

        if (shape.Rank == 0)
        {
            parameters[0] -= Step(gradients[0], ref state.Mean[0], ref state.Variance[0], time);
            return;
        }

        ReadOnlySpan<int> lengths = shape.Lengths;
        Span<int> indices = stackalloc int[shape.Rank];
        TensorShape.InitializeForwardIndexing(indices);

        while (TensorShape.MoveToNextIndex(lengths, indices))
        {
            parameters[indices] -= Step(gradients[indices], ref state.Mean[indices], ref state.Variance[indices], time);
        }
    }

    private TScalar Step(TScalar gradient, ref TScalar mean, ref TScalar variance, TScalar time)
    {
        mean = _beta1 * mean + (TScalar.One - _beta1) * gradient;
        variance = _beta2 * variance + (TScalar.One - _beta2) * gradient * gradient;

        TScalar beta1Pow = TScalar.Pow(_beta1, time);
        TScalar beta2Pow = TScalar.Pow(_beta2, time);

        TScalar meanDenom = TScalar.One - beta1Pow;
        TScalar varianceDenom = TScalar.One - beta2Pow;

        meanDenom = (meanDenom == TScalar.Zero) ? TScalar.Epsilon : meanDenom;
        varianceDenom = (varianceDenom == TScalar.Zero) ? TScalar.Epsilon : varianceDenom;

        TScalar meanHat = mean / meanDenom;
        TScalar varianceHat = variance / varianceDenom;

        return _learningRate * meanHat / (TScalar.Sqrt(varianceHat) + TScalar.Epsilon);
    }

    public void Clean()
    {
        _states.Clear();
    }

    private struct AdamState
    {
        public readonly Tensor<TScalar> Mean;
        public readonly Tensor<TScalar> Variance;
        public int Time;

        public AdamState(TensorShape shape)
        {
            Mean = Tensor.Create<TScalar>(shape);
            Variance = Tensor.Create<TScalar>(shape);
            Time = 0;
        }
    }
}
