using NeuralNetworks.Core;

namespace NeuralNetworks.UnitTests;

public sealed class PerceptronTests
{
    [Test]
    [Arguments<int>(1), Arguments<int>(1000)]
    public async ValueTask Constructor_WithPositiveLength_SetsLength(int length)
    {
        var perceptron = new Perceptron<float>(length);

        await Assert.That(perceptron.Length).IsEqualTo(length);
    }

    [Test]
    [Arguments<int>(-1), Arguments<int>(0)]
    public async ValueTask Constructor_WithNegativeOrZeroLength_Throws(int length)
    {
        var unit = () => new Perceptron<float>(length);

        await Assert.That(unit).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async ValueTask Train_WithNOTGateData_PredictsNOTGate()
    {
        float[][] inputData = [[0.0f], [1.0f]];
        float[] outputData = [1.0f, 0.0f];

        var perceptron = new Perceptron<float>(1);
        perceptron.Train(inputData, outputData, learningRate: 0.1f, epochs: 100);

        for (int dataIndex = 0; dataIndex < 2; ++dataIndex)
        {
            await Assert.That(perceptron.Predict(inputData[dataIndex])).IsEqualTo(outputData[dataIndex]);
        }
    }

    [Test]
    public async ValueTask Train_WithANDGateData_PredictsANDGate()
    {
        float[][] inputData = [[0.0f, 0.0f], [0.0f, 1.0f], [1.0f, 0.0f], [1.0f, 1.0f]];
        float[] outputData = [0.0f, 0.0f, 0.0f, 1.0f];

        var perceptron = new Perceptron<float>(2);
        perceptron.Train(inputData, outputData, learningRate: 0.1f, epochs: 100);

        for (int dataIndex = 0; dataIndex < 4; ++dataIndex)
        {
            await Assert.That(perceptron.Predict(inputData[dataIndex])).IsEqualTo(outputData[dataIndex]);
        }
    }

    [Test]
    public async ValueTask Train_WithORGateData_PredictsORGate()
    {
        float[][] inputData = [[0.0f, 0.0f], [0.0f, 1.0f], [1.0f, 0.0f], [1.0f, 1.0f]];
        float[] outputData = [0.0f, 1.0f, 1.0f, 1.0f];

        var perceptron = new Perceptron<float>(2);
        perceptron.Train(inputData, outputData, learningRate: 0.1f, epochs: 100);

        for (int dataIndex = 0; dataIndex < 4; ++dataIndex)
        {
            await Assert.That(perceptron.Predict(inputData[dataIndex])).IsEqualTo(outputData[dataIndex]);
        }
    }

    [Test]
    public async ValueTask Train_WithMismatchedBatchLengths_Throws()
    {
        float[][] inputData = [[0.0f], [1.0f]];
        float[] outputData = [1.0f];

        var perceptron = new Perceptron<float>(1);
        var unit = () => perceptron.Train(inputData, outputData, learningRate: 0.1f, epochs: 100);

        await Assert.That(unit).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    [Arguments<float>(-1.0f), Arguments<float>(0.0f)]
    public async ValueTask Train_WithNegativeOrZeroLearningRate_Throws(float learningRate)
    {
        float[][] inputData = [[0.0f], [1.0f]];
        float[] outputData = [1.0f, 0.0f];

        var perceptron = new Perceptron<float>(1);
        var unit = () => perceptron.Train(inputData, outputData, learningRate, epochs: 100);

        await Assert.That(unit).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    [Arguments<int>(-1), Arguments<int>(0)]
    public async ValueTask Train_WithNegativeOrZeroEpochs_Throws(int epochs)
    {
        float[][] inputData = [[0.0f], [1.0f]];
        float[] outputData = [1.0f, 0.0f];

        var perceptron = new Perceptron<float>(1);
        var unit = () => perceptron.Train(inputData, outputData, learningRate: 0.1f, epochs);

        await Assert.That(unit).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async ValueTask Predict_WithMismatchedDataLength_Throws()
    {
        float[] inputData = [0.0f];

        var perceptron = new Perceptron<float>(2);
        var unit = () => perceptron.Predict(inputData);

        await Assert.That(unit).Throws<ArgumentOutOfRangeException>();
    }

    [Test]
    public async ValueTask Train_WithLinearlySeparableData_ImprovesPredictionCorrectness()
    {
        float[][] inputData = [[0.0f, 0.0f], [0.0f, 1.0f], [1.0f, 0.0f], [1.0f, 1.0f]];
        float[] outputData = [0.0f, 0.0f, 0.0f, 1.0f];

        var perceptron = new Perceptron<float>(2);

        var calculateCorrectness = () => inputData.Zip(outputData)
            .Select((data) => perceptron.Predict(data.First) == data.Second).Count();

        int correctnessBefore = calculateCorrectness();
        perceptron.Train(inputData, outputData, learningRate: 0.1f, epochs: 100);
        int correctnessAfter = calculateCorrectness();

        await Assert.That(correctnessAfter).IsGreaterThanOrEqualTo(correctnessBefore)
            .And.IsEqualTo(inputData.Length);
    }
}
