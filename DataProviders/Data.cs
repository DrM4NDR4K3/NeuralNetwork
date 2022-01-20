using System;
using System.Collections.Generic;
using System.Text;

namespace DataProviders
{
    public class Data
    {
        public double[,] Inputs { get; }
        public double[,] Outputs { get; }

        public Data(double[,] inputs, double[,] outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }
    }
}
