using System.Numerics;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.Core.Connections;

public interface IConnection<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    Tensor<TScalar> Input { get; }
    Tensor<TScalar> Output { get; }

    void Calculate();

    Tensor<TScalar> Optimize(IOptimizer<TScalar> optimizer, in TensorSpan<TScalar> outputGradients);
}
