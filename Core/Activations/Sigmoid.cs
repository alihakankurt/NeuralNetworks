using System.Numerics;

namespace NeuralNetworks.Core.Activations;

public readonly struct Sigmoid<TScalar> : IActivation<TScalar>
    where TScalar : struct, IFloatingPointIeee754<TScalar>
{
    public static TScalar Activate(TScalar value)
    {
        return TScalar.One / (TScalar.One + TScalar.Exp(-value));
    }

    public static TScalar Derivate(TScalar value)
    {
        TScalar activation = Activate(value);
        return activation * (TScalar.One - activation);
    }
}
