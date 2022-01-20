using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Common.Serialization;
using NeuralNetwork.Layers;
using System;

namespace NeuralNetwork.Serialization
{
    internal static class LayerDeserializer
    {
        public static ILayer Deserialize(ISerializedLayer serializedLayer, int batchSize)
        {
            switch (serializedLayer.Type)
            {
                case LayerType.Standard:
                    var standardSerialized = serializedLayer as SerializedStandardLayer;
                    return DeserializeBasicStandardLayer(standardSerialized, batchSize);
                
                default:
                    throw new InvalidOperationException("Unknown layer type to deserialize");
            }
        }


        private static ILayer DeserializeBasicStandardLayer(SerializedStandardLayer standardSerialized, int batchSize)
        {
            var weights = Matrix<double>.Build.DenseOfArray(standardSerialized.Weights);
            var bias = Matrix<double>.Build.DenseOfColumnArrays(new double[][] { standardSerialized.Bias });
            var inputSize = weights.RowCount;
            var layerSize = weights.ColumnCount;
            var activator = ActivatorFactory.Build(standardSerialized.ActivatorType);
            return new BasicStandardLayer(weights, bias, batchSize, activator);
        }
    }
}