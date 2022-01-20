using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace DataProviders
{
    public class MathData
    {
        public MathData(Matrix<double> inputs, Matrix<double> outputs)
        {
            Inputs = inputs;
            Outputs = outputs;
        }

        public Matrix<double> Inputs { get; }
        public Matrix<double> Outputs { get; }
    }
}
