using System.Numerics;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.Core.Losses;

public interface ILoss<TScalar>
    where TScalar : struct, INumber<TScalar>
{
    static abstract TScalar Calculate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs);

    static abstract Tensor<TScalar> Derivate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs);

    static abstract void Derivate(in TensorSpan<TScalar> outputs, in TensorSpan<TScalar> expectedOutputs, in TensorSpan<TScalar> gradients);
}
