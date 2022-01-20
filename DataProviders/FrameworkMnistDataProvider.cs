using MNIST.IO;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DataProviders
{
    public class FrameworkMnistDataProvider : IDataProvider
    {
        private const string pathToFiles = @"D:\Dropbox\TravailMnacho\Enseignements\DeepLearningPricer\Sources\NeuralNetwork\MNIST\";
        private const string testLabels = @"t10k-labels-idx1-ubyte.gz";
        private const string testImages = @"t10k-images-idx3-ubyte.gz";
        private const string trainingLabels = @"train-labels-idx1-ubyte.gz";
        private const string trainingImages = @"train-images-idx3-ubyte.gz";
        public SplitData GetData()
        {
            var trainingData = ReadMnistData(pathToFiles + trainingLabels, pathToFiles + trainingImages);
            var testData = ReadMnistData(pathToFiles + testLabels, pathToFiles + testImages);
            return new SplitData(trainingData, testData);
        }

        private Data ReadMnistData(string pathToLabels, string pathToImages)
        {
            var trainingImageLabels = FileReaderMNIST.LoadImagesAndLables(pathToLabels, pathToImages).ToList();
            var trainingSampleSize = trainingImageLabels.Count;
            var firstTestCase = trainingImageLabels.First().AsDouble();
            var lineNb = firstTestCase.GetLength(0);
            var colNb = firstTestCase.GetLength(1);
            var trainingInputs = new double[lineNb * colNb, trainingSampleSize];
            var trainingOutputs = new double[1, trainingSampleSize];
            for (int tstNb = 0; tstNb < trainingSampleSize; tstNb++)
            {
                var currentSample = trainingImageLabels[tstNb];
                for (int i = 0; i < lineNb; i++)
                {
                    for (int j = 0; j < colNb; j++)
                    {
                        trainingInputs[i * lineNb + j, tstNb] = currentSample.Image[i, j];
                    }
                }
                trainingOutputs[0, tstNb] = trainingImageLabels[tstNb].Label;
            }
            var trainingData = new Data(trainingInputs, trainingOutputs);
            return trainingData;
        }
    }
}
