using NeuralNetworks.Core.Connections;
using NeuralNetworks.Core.Losses;
using NeuralNetworks.Core.Numerics;
using NeuralNetworks.Core.Optimizers;

namespace NeuralNetworks.DigitRecognizer;

public sealed class Model
{
    private const int InputLength = 784;
    private const int HiddenLength = 64;
    private const int OutputLength = 10;

    private const int TrainSamples = 60000;
    private const int TestSamples = 10000;

    private const int Epochs = 30;
    private const int BatchSize = 20;
    private const double LearningRate = 0.001f;

    private readonly Tensor<double> _layer1;
    private readonly Tensor<double> _layer2;
    private readonly Tensor<double> _layer3;
    private readonly Tensor<double> _layer4;
    private readonly Tensor<double> _layer5;

    private readonly IConnection<double> _connection1;
    private readonly IConnection<double> _connection2;
    private readonly IConnection<double> _connection3;
    private readonly IConnection<double> _connection4;

    private readonly IOptimizer<double> _optimizer;

    public Model()
    {
        _layer1 = Tensor.Create<double>([InputLength]);
        _layer2 = Tensor.Create<double>([HiddenLength]);
        _layer3 = Tensor.Create<double>([HiddenLength]);
        _layer4 = Tensor.Create<double>([OutputLength]);
        _layer5 = Tensor.Create<double>([OutputLength]);

        double inputStddev = double.Sqrt(2.0f / InputLength);
        double inputLimit = 2 * inputStddev;

        double hiddenStddev = double.Sqrt(2.0f / HiddenLength);
        double hiddenLimit = 2 * hiddenStddev;

        _connection1 = new Dense<double>(_layer1, _layer2, -inputLimit, inputLimit, -inputLimit, inputLimit);
        _connection2 = new ReLU<double>(_layer2, _layer3);
        _connection3 = new Dense<double>(_layer3, _layer4, -hiddenLimit, hiddenLimit, -hiddenLimit, hiddenLimit);
        _connection4 = new Softmax<double>(_layer4, _layer5);

        _optimizer = new Adam<double>(LearningRate);
    }

    public void Train()
    {
        var (trainInputs, trainOutputs) = DatasetReader.ReadTrainData(maxSamples: TrainSamples);
        int trainSampleCount = trainInputs.Count;
        Console.WriteLine($"Loaded train data: {trainSampleCount} samples.");

        int batchCount = trainSampleCount / BatchSize;
        if (batchCount * BatchSize != trainSampleCount)
        {
            Console.WriteLine("The batch size does not divide train samples exactly!");
        }

        Span<int> indices = stackalloc int[trainSampleCount];
        for (int index = 0; index < trainSampleCount; ++index)
        {
            indices[index] = index;
        }

        TensorSpan<double> layer1Span = _layer1.AsSpan();
        TensorSpan<double> layer5Span = _layer5.AsSpan();

        var gradients4 = Tensor.Create<double>([OutputLength]);
        TensorSpan<double> gradients4Span = gradients4.AsSpan();

        for (int epoch = 1; epoch <= Epochs; ++epoch)
        {
            if (epoch % (int.Max(Epochs / 100, 1)) == 0)
            {
                Console.WriteLine($"Training: {epoch}/{Epochs} ({(double)epoch / Epochs:P2}) epochs");
            }

            double totalLoss = 0, totalBatchLoss;
            Random.Shared.Shuffle(indices);
            for (int batchIndex = 0; batchIndex < batchCount; ++batchIndex)
            {
                totalBatchLoss = 0;

                gradients4.Fill(0);
                for (int i = 0; i < BatchSize; ++i)
                {
                    int index = indices[batchIndex * BatchSize + i];

                    TensorSpan<double> inputSpan = trainInputs[index].AsSpan();
                    TensorSpan<double> outputSpan = trainOutputs[index].AsSpan();

                    layer1Span.Set(inputSpan);

                    _connection1.Calculate();
                    _connection2.Calculate();
                    _connection3.Calculate();
                    _connection4.Calculate();

                    double loss = CrossEntropyLoss<double>.Calculate(layer5Span, outputSpan);
                    totalBatchLoss += loss;

                    CrossEntropyLoss<double>.Derivate(layer5Span, outputSpan, gradients4Span);
                }

                Tensor.Divide(gradients4Span, BatchSize);

                var gradients3 = _connection4.Optimize(_optimizer, gradients4Span);
                var gradients2 = _connection3.Optimize(_optimizer, gradients3.AsSpan());
                var gradients1 = _connection2.Optimize(_optimizer, gradients2.AsSpan());
                _connection1.Optimize(_optimizer, gradients1.AsSpan());

                totalLoss += totalBatchLoss / BatchSize;
            }

            Console.WriteLine($"Loss: {totalLoss / batchCount}");
        }

        _optimizer.Clean();

        Console.WriteLine("Training complete.");
    }

    public void Test()
    {
        var (testInputs, testOutputs) = DatasetReader.ReadTestData(maxSamples: TestSamples);
        int testSampleCount = testInputs.Count;
        Console.WriteLine($"Loaded test data: {testSampleCount} samples.");

        int correctPredictions = 0;
        double lossSum = 0f;
        var confusionMatrix = new int[OutputLength, OutputLength];
        var outputSpan = _layer5.AsSpan();

        for (int testIndex = 0; testIndex < testSampleCount; ++testIndex)
        {
            if ((testIndex + 1) % (int.Max(1, testSampleCount / 100)) == 0)
            {
                Console.WriteLine($"Testing: {(testIndex + 1)}/{testSampleCount} ({(double)(testIndex + 1) / testSampleCount:P2})");
            }

            _layer1.Set(testInputs[testIndex]);
            _connection1.Calculate();
            _connection2.Calculate();
            _connection3.Calculate();
            _connection4.Calculate();

            int predictedDigit = 0;
            double predictedDigitRate = outputSpan[0];
            for (int digit = 1; digit < OutputLength; ++digit)
            {
                if (outputSpan[digit] > predictedDigitRate)
                {
                    predictedDigit = digit;
                    predictedDigitRate = outputSpan[digit];
                }
            }

            var testOutputSpan = testOutputs[testIndex].AsSpan();
            int correctDigit = 0;
            double correctDigitRate = testOutputSpan[0];
            for (int digit = 1; digit < OutputLength; ++digit)
            {
                if (testOutputSpan[digit] > correctDigitRate)
                {
                    correctDigit = digit;
                    correctDigitRate = testOutputSpan[digit];
                }
            }

            confusionMatrix[correctDigit, predictedDigit]++;
            if (predictedDigit == correctDigit)
                correctPredictions++;

            lossSum += CrossEntropyLoss<double>.Calculate(outputSpan, testOutputSpan);
        }

        Console.WriteLine("Testing complete.");
        Console.WriteLine("Test Results: ");

        double accuracy = (double)correctPredictions / testSampleCount;
        double averageLoss = lossSum / testSampleCount;

        double macroPrecision = 0, macroRecall = 0, macroF1 = 0;
        Console.WriteLine("Digit | Precision | Recall | F1");
        for (int digit = 0; digit < OutputLength; ++digit)
        {
            int tp = confusionMatrix[digit, digit];
            int fp = 0, fn = 0;
            for (int other = 0; other < OutputLength; ++other)
            {
                if (other != digit)
                {
                    fp += confusionMatrix[other, digit];
                    fn += confusionMatrix[digit, other];
                }
            }

            double precision = (tp + fp == 0) ? 0 : tp / (double)(tp + fp);
            double recall = (tp + fn == 0) ? 0 : tp / (double)(tp + fn);
            double f1 = (precision + recall == 0) ? 0 : 2 * precision * recall / (double)(precision + recall);

            macroPrecision += precision;
            macroRecall += recall;
            macroF1 += f1;

            Console.WriteLine($"{digit,5} | {precision,9:P2} | {recall,6:P2} | {f1,6:P2}");
        }

        macroPrecision /= OutputLength;
        macroRecall /= OutputLength;
        macroF1 /= OutputLength;

        Console.WriteLine($"Test accuracy: {accuracy:P2}");
        Console.WriteLine($"Average loss: {averageLoss:F4}");
        Console.WriteLine($"Macro Precision: {macroPrecision:P2}");
        Console.WriteLine($"Macro Recall: {macroRecall:P2}");
        Console.WriteLine($"Macro F1: {macroF1:P2}");
        Console.WriteLine();
    }
}
