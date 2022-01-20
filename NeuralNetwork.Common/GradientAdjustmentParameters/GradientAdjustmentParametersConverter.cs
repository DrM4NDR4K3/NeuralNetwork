using NeuralNetwork.Common.GradientAdjustmentParameters;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

namespace NeuralNetwork.Common.GradientAdjustmentParameters
{
    /// <summary>
    /// Converter for generating objects of type <see cref="IGradientAdjustmentParameters"/> from a Json object.
    /// </summary>
    /// <seealso cref="Newtonsoft.Json.JsonConverter" />
    public class GradientAdjustmentParametersConverter : JsonConverter
    {
        public override bool CanWrite => false;
        public override bool CanRead => true;

        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(IGradientAdjustmentParameters);
        }

        public override void WriteJson(JsonWriter writer,
            object value, JsonSerializer serializer)
        {
            throw new InvalidOperationException("Use default serialization.");
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue, JsonSerializer serializer)
        {
            var jsonObject = JObject.Load(reader);
            IGradientAdjustmentParameters asset;
            var readType = jsonObject.GetValue("type", StringComparison.OrdinalIgnoreCase).Value<string>();
            var enumValue = Enum.Parse(typeof(GradientAdjustmentType), readType, true);
            switch (enumValue)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    asset = new FixedLearningRateParameters();
                    break;

                case GradientAdjustmentType.Adam:
                    asset = new AdamParameters();
                    break;

                case GradientAdjustmentType.Momentum:
                    asset = new MomentumParameters();
                    break;

                default:
                    throw new InvalidOperationException("Unknown gradient accelerator parameter");
            }
            serializer.Populate(jsonObject.CreateReader(), asset);
            return asset;
        }
        
    }
}