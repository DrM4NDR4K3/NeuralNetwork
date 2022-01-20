using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using System;

namespace NeuralNetwork.Layers
{
    internal class BasicStandardLayer : ILayer
    {

        public int InputSize { get; }
        public int LayerSize { get; }
        public Matrix<double> InitialBias { get; set; }      
        public Matrix<double> InitialWeights { get; set; }

        public int BatchSize { get; set; }

        public IActivator Activator { get; }


        public Matrix<double> Activation { get; }

        public Matrix<double> WeightedError { get; }

        public BasicStandardLayer(Matrix<double> initialWeights, Matrix<double> initialBias, int batchSize, IActivator activator)
        {
            InitialBias = initialBias;
            InitialWeights = initialWeights;
            LayerSize = initialWeights.ColumnCount;
            InputSize = initialWeights.RowCount;
            BatchSize = batchSize;
            Activator = activator ?? throw new ArgumentNullException(nameof(activator));
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
        }

        public void Propagate(Matrix<double> input)
        {
            Matrix<double> Mat = InitialBias.Add(InitialWeights.Transpose().Multiply(input));
            Mat.Map(Activator.Apply, this.Activation);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            throw new NotImplementedException();
        }

        public void UpdateParameters()
        {
            throw new NotImplementedException();
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}