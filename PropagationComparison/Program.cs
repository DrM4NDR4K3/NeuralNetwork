using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Serialization;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace PropagationComparison
{
    internal static class Program
    {
        static void Main(string[] args)
        {
            var serializedNetworkPath = args[0];
            var serializedNetwork = JsonConvert.DeserializeObject<SerializedNetwork>(File.ReadAllText(serializedNetworkPath));
            var network = NetworkDeserializer.Deserialize(serializedNetwork);
            var inputRoot = args[1];
            var gradientRoot = args[2];
            var data = GetMathData(inputRoot, gradientRoot);
            var dataSize = data.Inputs.ColumnCount;
            network.BatchSize = dataSize;
            network.Propagate(data.Inputs);
            double[] firstPropagation = ConvertToArray(network.Output);
            network.Learn(data.Outputs);
            network.Propagate(data.Inputs);
            double[] secondPropagation = ConvertToArray(network.Output);
            var summary = new OutputSummary(firstPropagation, secondPropagation);
            if (args.Length < 4)
            {
                WriteToConsole(summary);
            }
            else
            {
                var outputFile = args[3];
                WriteToFile(summary, outputFile);
            }

        }


        private static void WriteToConsole(OutputSummary summary)
        {
            var output = JsonConvert.SerializeObject(summary, Formatting.Indented);
            Console.WriteLine(output);
        }

        private static void WriteToFile(OutputSummary summary, string outputFile)
        {
            JsonSerializer serializer = new JsonSerializer();
            using (StreamWriter sw = new StreamWriter(outputFile))
            using (JsonWriter writer = new JsonTextWriter(sw))
            {
                serializer.Serialize(writer, summary);
            }
        }
        private static double[] ConvertToArray(Matrix<double> output)
        {
            return output.ToRowArrays()[0];
        }

        private static MathData GetMathData(string dataRoot, string gradientEoot)
        {
            var input = ReadCsv(dataRoot);
            var inputMatrix = DenseMatrix.OfColumnArrays(input);
            var output = ReadCsv(gradientEoot);
            var outputMatrix = DenseMatrix.OfColumnArrays(output);
            var result = new MathData(inputMatrix, outputMatrix);
            return result;
        }

        private static IEnumerable<double[]> ReadCsv(string filePath)
        {
            CultureInfo customCulture = (CultureInfo)System.Threading.Thread.CurrentThread.CurrentCulture.Clone();
            customCulture.NumberFormat.NumberDecimalSeparator = ".";
            System.Threading.Thread.CurrentThread.CurrentCulture = customCulture;
            using (StreamReader sr = new StreamReader(filePath))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    double[] arr = line.Split(',').Select(double.Parse).ToArray();
                    yield return arr;
                }
            }
        }
    }
}
