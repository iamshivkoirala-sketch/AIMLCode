using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public class InsuranceData
    {
        [LoadColumn(0)]
        public float Age { get; set; }
        [LoadColumn(1)]
        public float Premium { get; set; }
    }
    public class InsurancePrediction
    {
        [ColumnName("Score")]
        public float PredictedPremium { get; set; }
    }
}
