using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.Data.Analysis;

using Microsoft.ML;
using Microsoft.ML.Runtime;
using Parquet;
using Parquet.File;
using ParquetSharp.RowOriented;
using Parquet.Data;
using System.Data;
namespace MachinelearningClass
{
    public   class GuardRailsExample
    {
        static MLContext context1 = null;
        static PredictionEngine<ModelInput, ModelOutput> pred = null;
        public static List<ModelInput> ReadAllBudData()
        {
            var observations = new List<ModelInput>();


            using (var reader = ParquetFile.CreateRowReader<ParquetInput>(Program.datapath + "train-00000-of-00004.parquet"))
            {
                int numRowGroups = reader.FileMetaData.NumRowGroups;

                for (int i = 0; i < numRowGroups; i++)
                {
                    // Read rows as ParquetInput (handles the nulls/bool?)
                    ParquetInput[] groupRows = reader.ReadRows(i);

                    foreach (var row in groupRows)
                    {
                        if (row.text != null && row.is_safe.HasValue)
                        {
                            observations.Add(new ModelInput
                            {
                                Text = row.text,
                                Label = row.is_safe.Value 
                            });
                        }
                    }
                }
            }
            return observations;
        }
        // static GuardRailsExample()
        //{
        //    var context = new MLContext();


        //    //var data = context.Data.LoadFromTextFile<ModelInput>(Program.datapath +  "guardrail.csv", hasHeader: true, separatorChar: ',');
        //    //var data = context.Data.LoadFromEnumerable(ReadAllBudData());
        //     var data = context.Data.LoadFromTextFile<ModelInput>(Program.datapath + "guardrail.csv", hasHeader: true, separatorChar: ',');

        //    var pipeline = context.Transforms.Text.FeaturizeText("Features", "Text")
        //        .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

        //    var model = pipeline.Fit(data);

        //    var pred = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
        //}
        public static bool Check(string prompt)
        {
            var result = pred.Predict(new ModelInput { Text = prompt });
            return result.IsSafe;
        }
        public static void Load()
        {
            var context = new MLContext();


            //var data = context.Data.LoadFromTextFile<ModelInput>(Program.datapath +  "guardrail.csv", hasHeader: true, separatorChar: ',');
           var data = context.Data.LoadFromEnumerable(ReadAllBudData());
           //var data = context.Data.LoadFromTextFile<ModelInput>(Program.datapath + "guardrail.csv", hasHeader: true, separatorChar: ',');
            
            var pipeline = context.Transforms.Text.FeaturizeText("Features", "Text")
                .Append(context.BinaryClassification.Trainers.SdcaLogisticRegression());

            var model = pipeline.Fit(data);

            var engine = context.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
            while (true)
            {
                Console.WriteLine("Enter a prompt");
                var testPrompt = Console.ReadLine();
                var result = engine.Predict(new ModelInput { Text = testPrompt });

                Console.WriteLine($"Is it safe? {result.IsSafe}");
            }
        }
    }
    public class ParquetInput
    {
        public string? text { get; set; }
        public bool? is_safe { get; set; }
    }
    public class ModelInput { [LoadColumn(0)] public string Text; 
                              [LoadColumn(1)] public bool Label; }

    public class ModelOutput { [ColumnName("PredictedLabel")] public bool IsSafe; }
}
