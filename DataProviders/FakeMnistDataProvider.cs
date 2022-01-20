using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DataProviders
{
    public class FakeMnistDataProvider : IDataProvider
    {
        private const string pathToFiles = @"D:\Dropbox\TravailMnacho\Enseignements\DeepLearningPricer\Sources\NeuralNetwork\MNIST\";
        private const string testDataPath = @"mnist_test.csv";
        private const string trainingDataPath = @"mnist_train.csv";

        public SplitData GetData()
        {
            var trainingData = MakeSparseRepresentation(pathToFiles + trainingDataPath);
            var testData = MakeSparseRepresentation(pathToFiles + testDataPath);
            return new SplitData(trainingData, testData);
        }

        private MathData MakeSparseRepresentation(string path)
        {
            var trainingData = ReandMnistData(path);
            SparseMatrix readMatrix = SparseMatrix.OfColumnArrays(trainingData.Take(10));
            var inputSize = readMatrix.RowCount;
            var dataSize = readMatrix.ColumnCount;
            //var outputs = readMatrix.SubMatrix(0, 1, 0, dataSize);
            var inputs = readMatrix.SubMatrix(1, inputSize - 1, 0, dataSize);
            inputs.MapInplace(el => (el / 255) * 0.99 + 0.01);
            var outputs = Matrix.Build.Dense(10, dataSize, 0.01);
            for (int sample = 0; sample < dataSize; sample++)
            {
                outputs[(int)readMatrix[0, sample], sample] = 0.99;
            }
            return new MathData(inputs, outputs);
        }


        public IEnumerable<double[]> ReandMnistData(string pathToData)
        {
            using (StreamReader sr = new StreamReader(pathToData))
            {
                string line;
                while ((line = sr.ReadLine()) != null)
                {
                    double[] arr = line.Split(',').Select(Double.Parse).ToArray();
                    yield return arr;
                }

            }
        }
    }
}
