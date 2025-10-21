using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Connections;

public sealed class ReLU<TScalar> : ElementwiseConnectionBase<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    public ReLU(Tensor<TScalar> input, Tensor<TScalar> output) : base(input, output)
    {
    }

    protected override TScalar Calculate(TScalar value)
    {
        return TScalar.Max(TScalar.Zero, value);
    }

    protected override TScalar Derivate(TScalar value)
    {
        return (value < TScalar.Zero) ? TScalar.Zero : TScalar.One;
    }
}
