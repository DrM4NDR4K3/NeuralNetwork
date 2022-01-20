using NeuralNetwork.Common.Activators;
using System;

namespace NeuralNetwork.Activators
{
    public class Identity : IActivator
    {
        public Func<double, double> Apply => (x) => x;

        public Func<double, double> ApplyDerivative => (x) => 1;

        public ActivatorType Type => ActivatorType.Identity;
    }
}