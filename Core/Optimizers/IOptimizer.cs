using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Optimizers;

public interface IOptimizer<TScalar> : IDisposable
    where TScalar : struct, INumber<TScalar>
{
    void Optimize(Tensor<TScalar> parameters, in TensorSpan<TScalar> gradients);
}
