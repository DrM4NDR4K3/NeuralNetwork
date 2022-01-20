using NeuralNetwork.Common.Layers;
using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace NeuralNetwork.Common.Serialization
{
    /// <summary>
    /// Interface for defining serialized layers, that can be written to, e.g., Json files.
    /// </summary>
    [JsonConverter(typeof(SerializedLayerConverter))]
    public interface ISerializedLayer
    {
        [JsonConverter(typeof(StringEnumConverter))]
        LayerType Type { get; }
    }
}