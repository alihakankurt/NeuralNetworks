using System.Numerics;

namespace NeuralNetworks.Core;

/// <summary>
/// Represents the simplest artificial neural network with single binary output.
/// </summary>
public class Perceptron<TScalar> where TScalar : struct, IBinaryFloatingPointIeee754<TScalar>
{
    /// <summary>The connection weights that is used to calculate the linear output by multiplying with the respective input values.</summary>
    private readonly TScalar[] _weights;

    /// <summary>The value that directly effects the linear output.</summary>
    private TScalar _bias;

    /// <summary>Gets the number of input neurons.</summary>
    public int Length { get; }

    /// <summary>
    /// Initializes a new instance of <see cref="Perceptron{TScalar}"/>.
    /// </summary>
    /// <param name="length">The number of input neurons.</param>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public Perceptron(int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);

        _weights = new TScalar[length];
        _bias = TScalar.Zero;
        Length = length;

        Randomize();
    }

    /// <summary>
    /// Trains itself using a batch of inputs and their expected outputs.
    /// </summary>
    /// <param name="inputBatch">The collection of input values.</param>
    /// <param name="outputBatch">The expected output values for each input.</param>
    /// <param name="learningRate">The speed of learning which effects the changes on weights and bias, must be between 0.0 and 1.0 inclusively.</param>
    /// <param name="epochs">The number of complete passes through the entire dataset, must be greater than 0.</param>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public void Train(ReadOnlySpan<TScalar[]> inputBatch, ReadOnlySpan<TScalar> outputBatch, TScalar learningRate, int epochs)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(learningRate, TScalar.Zero);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(learningRate, TScalar.One);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(epochs);
        ArgumentOutOfRangeException.ThrowIfNotEqual(inputBatch.Length, outputBatch.Length);

        int losses = 1;
        Span<int> indices = Enumerable.Range(0, inputBatch.Length).ToArray();

        while (epochs > 0 && losses > 0)
        {
            epochs--;
            losses = 0;
            Random.Shared.Shuffle(indices);

            foreach (var index in indices)
            {
                ReadOnlySpan<TScalar> inputs = inputBatch[index];
                TScalar expectedOutput = outputBatch[index];

                TScalar prediction = Predict(inputs);
                TScalar loss = prediction - expectedOutput;

                if (TScalar.Abs(loss) < TScalar.Epsilon)
                    continue;

                losses++;
                TScalar gradient = learningRate * loss;

                for (int neuron = 0; neuron < Length; ++neuron)
                {
                    _weights[neuron] -= gradient * inputs[neuron];
                }

                _bias -= gradient;
            }
        }
    }

    /// <summary>
    /// Predicts an output for a specified <paramref name="inputs"/>.
    /// </summary>
    /// <param name="inputs">The input values.</param>
    /// <exception cref="ArgumentOutOfRangeException"/>
    public TScalar Predict(ReadOnlySpan<TScalar> inputs)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(inputs.Length, Length);

        TScalar linearOutput = TScalar.Zero;

        for (int neuron = 0; neuron < Length; ++neuron)
        {
            linearOutput += inputs[neuron] * _weights[neuron];
        }

        linearOutput += _bias;

        TScalar prediction = (linearOutput >= TScalar.Zero) ? TScalar.One : TScalar.Zero;
        return prediction;
    }

    /// <summary>
    /// Randomizes the weights and bias values between -1.0 and 1.0 to provide better results in training.
    /// </summary>
    private void Randomize()
    {
        var generate = () => TScalar.CreateTruncating(Random.Shared.NextDouble() * 2 - 1);

        for (int neuron = 0; neuron < Length; ++neuron)
        {
            _weights[neuron] = generate();
        }

        _bias = generate();
    }
}
