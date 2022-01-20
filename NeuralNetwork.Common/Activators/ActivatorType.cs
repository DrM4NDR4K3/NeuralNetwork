using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace NeuralNetwork.Common.Activators
{
    /// <summary>
    /// Types of activators.
    /// </summary>
    [JsonConverter(typeof(StringEnumConverter))]
    public enum ActivatorType
    {
        Identity,
        Sigmoid,
        Tanh,
        LeakyReLU,
        ReLU
    }
}