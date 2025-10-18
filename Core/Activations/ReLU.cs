using System.Numerics;

namespace NeuralNetworks.Core.Activations;

public readonly struct ReLU<TScalar> : IActivation<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    public static TScalar Activate(TScalar value)
    {
        return TScalar.Max(TScalar.Zero, value);
    }

    public static TScalar Derivate(TScalar value)
    {
        return (value < TScalar.Zero) ? TScalar.Zero : TScalar.One;
    }
}
