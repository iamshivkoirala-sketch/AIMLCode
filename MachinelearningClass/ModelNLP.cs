using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass.ModelNLP
{
    public class FruitData
    {
        public string Fruit { get; set; }
    }

    public class FruitFeatures
    {
        public string Fruit { get; set; } = string.Empty;

        // [VectorType]
        public float[] FruitEncoded { get; set; }
    }
    public class InputText
    {
        [LoadColumn(0)]
        public string Text { get; set; }
    }

    public class TextFeatures
    {
        [ColumnName("Features")]
        [VectorType]
        public float[] Features { get; set; }
    }
}
