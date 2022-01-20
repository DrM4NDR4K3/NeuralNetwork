using NeuralNetwork.Common.Layers;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using System;

namespace NeuralNetwork.Common.Serialization
{
    /// <summary>
    /// Converter for generating objects of type <see cref="ISerializedLayer"/> from a Json object.
    /// </summary>
    /// <seealso cref="Newtonsoft.Json.JsonConverter" />
    public class SerializedLayerConverter : JsonConverter
    {
        public override bool CanWrite => false;
        public override bool CanRead => true;

        public override bool CanConvert(Type objectType)
        {
            return objectType == typeof(ISerializedLayer);
        }

        public override void WriteJson(JsonWriter writer,
            object value, JsonSerializer serializer)
        {
            throw new InvalidOperationException("Use default serialization.");
        }

        public override object ReadJson(JsonReader reader, Type objectType, object existingValue,  JsonSerializer serializer)
        {
            var jsonObject = JObject.Load(reader);
            ISerializedLayer asset;
            var readType = jsonObject.GetValue("type", StringComparison.OrdinalIgnoreCase).Value<string>();
            var readEnum = Enum.Parse(typeof(LayerType), readType, true);
            switch (readEnum)
            {
                case LayerType.Standard:
                    asset = new SerializedStandardLayer();
                    break;

                case LayerType.InputStandardizing:
                    asset = new SerializedInputStandardizingLayer();
                    break;

                case LayerType.Dropout:
                    asset = new SerializedDropoutLayer();
                    break;
                case LayerType.L2Penalty:
                    asset = new SerializedL2PenaltyLayer();
                    break;
                case LayerType.WeightDecay:
                    asset = new SerializedWeightDecayLayer();
                    break;
                default:
                    throw new InvalidOperationException("Unknown serialized layer: " + readType);
            }
            serializer.Populate(jsonObject.CreateReader(), asset);
            return asset;
        }

    }
}