using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Connections;

public sealed class Sigmoid<TScalar> : ElementwiseConnectionBase<TScalar>
    where TScalar : struct, IFloatingPointIeee754<TScalar>
{
    public Sigmoid(Tensor<TScalar> input, Tensor<TScalar> output) : base(input, output)
    {
    }

    protected override TScalar Calculate(TScalar value)
    {
        return TScalar.One / (TScalar.One + TScalar.Exp(-value));
    }

    protected override TScalar Derivate(TScalar value)
    {
        TScalar activation = Calculate(value);
        return activation * (TScalar.One - activation);
    }
}
