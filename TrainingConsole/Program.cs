using DataProviders;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Serialization;
using Newtonsoft.Json;
using System.IO;
using Trainer;
using Trainer.CostFunctions;
using Trainer.DataShufflers;

namespace TrainingConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            var jsonSerializerSettings = new JsonSerializerSettings();
            jsonSerializerSettings.Converters.Add(new Newtonsoft.Json.Converters.StringEnumConverter());
            var serializedNetwork = JsonConvert.DeserializeObject<SerializedNetwork>(File.ReadAllText(args[0]), jsonSerializerSettings);
            var network = NetworkDeserializer.Deserialize(serializedNetwork);
            var trainParams = JsonConvert.DeserializeObject<TrainingParams>(File.ReadAllText(args[1]));
            var dataProvider = DataProviderConverter.Convert(trainParams.ProviderType);
            var data = dataProvider.GetData();
            var trainingData = data.TrainingData;
            var validationData = data.ValidationData;
            IDataShuffler dataShuffler = trainParams.Shuffle ? new UniformShuffle() : new NoShuffle();
            var runner = new NetworkTrainer(network, new QuadraticError(), dataShuffler);
            for (int i = 0; i < trainParams.EpochNb; i++)
            {
                runner.Train(trainingData);
                if (i % trainParams.ValidationStep == 0)
                {
                    var currentValidationError = runner.Validate(validationData);
                    var currentTrainingError = runner.Validate(trainingData);
                    System.Console.WriteLine($"Epoch {i}. Training error {currentTrainingError} -- Validation error {currentValidationError}");
                }
            }
            var serializedContent = JsonConvert.SerializeObject(NetworkSerializer.Serialize(network), jsonSerializerSettings);
            File.WriteAllText(args[2], serializedContent);
            //var sample = new TrainingParams()
            //{
            //    EpochNb = 100,
            //    ProviderType = DataProviderType.AndData,
            //    Shuffle = true,
            //    ValidationStep = 10
            //};
            //var serializedContent = JsonConvert.SerializeObject(sample, jsonSerializerSettings);
            //File.WriteAllText("sample.json", serializedContent);
        }

        
    }
}
