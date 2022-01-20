using Newtonsoft.Json;
using Newtonsoft.Json.Converters;

namespace NeuralNetwork.Common.GradientAdjustmentParameters
{
    /// <summary>
    /// Interface for gradient adjustment parameters.
    /// </summary>
    [JsonConverter(typeof(GradientAdjustmentParametersConverter))]
    public interface IGradientAdjustmentParameters
    {
        [JsonConverter(typeof(StringEnumConverter))]
        GradientAdjustmentType Type { get; }
    }
}