using System.Buffers;
using System.Globalization;
using NeuralNetworks.Core.Numerics;

namespace NeuralNetworks.DigitRecognizer;

public static class DatasetReader
{
    private const string TrainFilePath = "./Datasets/MNIST/train.csv";
    private const string TestFilePath = "./Datasets/MNIST/test.csv";

    private const int PixelIndex = 0;
    private const int PixelCount = 28 * 28;

    private const int DigitIndex = PixelCount;
    private const int DigitCount = 10;

    private const int ValuesPerSample = PixelCount + DigitCount;

    public static (List<Tensor<double>> Inputs, List<Tensor<double>> Outputs) ReadTrainData(int maxSamples = int.MaxValue)
    {
        return DatasetReader.Read(TrainFilePath, maxSamples);
    }

    public static (List<Tensor<double>> Inputs, List<Tensor<double>> Outputs) ReadTestData(int maxSamples = int.MaxValue)
    {
        return DatasetReader.Read(TestFilePath, maxSamples);
    }

    private static (List<Tensor<double>> Inputs, List<Tensor<double>> Outputs) Read(string filePath, int maxSamples)
    {
        using var stream = File.OpenRead(filePath);
        using var reader = new StreamReader(stream);

        List<Tensor<double>> inputs = [];
        List<Tensor<double>> outputs = [];

        double[] bufferArray = ArrayPool<double>.Shared.Rent(ValuesPerSample);
        Span<double> buffer = bufferArray[..ValuesPerSample];

        try
        {
            while (!reader.EndOfStream && inputs.Count < maxSamples)
            {
                string? line = reader.ReadLine();
                if (string.IsNullOrEmpty(line))
                {
                    Console.WriteLine($"Error at line {inputs.Count + 1} in {filePath}. Skipping...");
                    continue;
                }

                int valueIndex = 0;
                ReadOnlySpan<char> chars = line;

                foreach (Range range in chars.Split(','))
                {
                    buffer[valueIndex++] = double.Parse(chars[range], provider: CultureInfo.InvariantCulture);
                }

                if (valueIndex != ValuesPerSample)
                {
                    Console.WriteLine($"Expected {ValuesPerSample} but found {valueIndex} values at line {inputs.Count + 1} in {filePath}. Skipping...");
                    continue;
                }

                var input = Tensor.CreateFrom<double>(buffer.Slice(PixelIndex, PixelCount));
                var output = Tensor.CreateFrom<double>(buffer.Slice(DigitIndex, DigitCount));

                inputs.Add(input);
                outputs.Add(output);
            }

            return (inputs, outputs);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(bufferArray);
        }
    }
}
