using System.Numerics;

namespace NeuralNetworks.Core.Activations;

public interface IActivation<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    static abstract TScalar Activate(TScalar value);
    static abstract TScalar Derivate(TScalar value);
}
