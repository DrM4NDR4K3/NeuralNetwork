using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace DataProviders
{
    public class MnistDataProvider : IDataProvider
    {

        private const string pathToFiles = @"D:\Dropbox\TravailMnacho\Enseignements\DeepLearningPricer\Sources\NeuralNetwork\MNIST\";
        private const string testDataPath = @"mnist_test.csv";
        private const string trainingDataPath = @"mnist_train.csv";

        public SplitData GetData()
        {
            var trainingData = ReandMnistData(pathToFiles + trainingDataPath);
            var testData = ReandMnistData(pathToFiles + testDataPath);
            return new SplitData(trainingData, testData);
        }

        public Data ReandMnistData(string pathToData)
        {
            using (StreamReader sr = new StreamReader(pathToData))
            {
                string line;
                List<int[]> datas = new List<int[]>();
                while ((line = sr.ReadLine()) != null)
                {
                    int[] arr = line.Split(',').Select(Int32.Parse).ToArray();
                    datas.Add(arr);
                }
                var sampleSize = datas.Count;
                var lineNb = datas.First().Length;
                var trainingInputs = new double[lineNb - 1, sampleSize];
                var trainingOutputs = new double[1, sampleSize];
                for (int tstNb = 0; tstNb < sampleSize; tstNb++)
                {
                    trainingOutputs[0, tstNb] = datas[tstNb][0];
                    for (int i = 0; i < lineNb - 1; i++)
                    {
                        trainingInputs[i, tstNb] = datas[tstNb][i + 1];

                    }
                }
                return new Data(trainingInputs, trainingOutputs);
            }
        }
    }
}
