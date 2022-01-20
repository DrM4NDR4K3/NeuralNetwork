using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork
{
    public sealed class Network : INetwork
    {
        private int batchSize;
        private Mode mode;

        public Mode Mode
        {
            get => mode;
            set
            {
                if (value != mode)
                {
                    mode = value;
                    UpdateLayersWithMode(value);
                }
            }
        }

        private void UpdateLayersWithMode(Mode mode)
        {
            for (int i = 0; i < LayerNb; i++)
            {
                if (Layers[i] is IComponentWithMode layerWithMode)
                {
                    layerWithMode.Mode = mode;
                }
            }
        }

        public ILayer[] Layers { get; }
        public int LayerNb { get; }

        public int BatchSize
        {
            get => batchSize;
            set
            {
                batchSize = value;
                foreach (var layer in Layers)
                {
                    layer.BatchSize = value;
                }
            }
        }

        internal ILayer OutputLayer => Layers[LayerNb - 1];
        public Matrix<double> Output => OutputLayer.Activation;

        public Network(ILayer[] layers, int batchSize)
        {
            Layers = layers ?? throw new ArgumentNullException(nameof(layers));
            if (Layers.Length == 0)
            {
                throw new InvalidOperationException("The network must contain at least one layer");
            }
            LayerNb = Layers.Length;
            BatchSize = batchSize;
        }

        public void Propagate(Matrix<double> input)
        {
            Layers[0].Propagate(input);
            for (int i = 1; i < LayerNb; i++)
            {
                Layers[i].Propagate(Layers[i - 1].Activation);
            }
        }

        public void Learn(Matrix<double> lossFunctionGradient)
        {
            BackpropAndUpdate(OutputLayer, lossFunctionGradient);
            for (int i = LayerNb - 2; i >= 0; i--)
            {
                BackpropAndUpdate(Layers[i], Layers[i + 1].WeightedError);
            }
        }

        private void BackpropAndUpdate(ILayer layer, Matrix<double> outputLayerError)
        {
            layer.BackPropagate(outputLayerError);
            layer.UpdateParameters();
        }
    }
}