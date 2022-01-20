using DataProviders;
using Newtonsoft.Json.Converters;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace TrainingConsole
{
    internal class TrainingParams
    {
        public int EpochNb { get; set; }
        
        public DataProviderType ProviderType { get; set; }
        public int ValidationStep { get; set; }
        public bool Shuffle { get; set; }
    }
}
