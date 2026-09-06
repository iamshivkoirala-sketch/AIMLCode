using MachinelearningClass.ModelNLP;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public class Week3 // NLP
    {
        public static void Lab11_OneHotEncoding()
        {
            var ml = new MLContext();

            var data = new[]
            {
            new FruitData { Fruit = "Mango" },
            new FruitData { Fruit = "Apple" },
            new FruitData { Fruit = "Berry" }
        };

            var dataView = ml.Data.LoadFromEnumerable(data);

            var pipeline = ml.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "FruitEncoded",
                inputColumnName: "Fruit");

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);

            var encoded = ml.Data.CreateEnumerable<FruitFeatures>(
                transformedData, reuseRowObject: false);

            Console.WriteLine("One-Hot Encoded Vectors:");
            foreach (var row in encoded)
            {
                Console.WriteLine($"[{string.Join(",", row.FruitEncoded)}]");
            }
        }
        public static void Lab12_Bow()
        {
            var mlContext = new MLContext();

                var samples = new[]
                {
                new InputText { Text = "This camera camera is good" },
                new InputText { Text = "This camera is bad" }
                };

            // 1. Still Fit on the whole array so the model knows 'good' AND 'bad'
            var dataView = mlContext.Data.LoadFromEnumerable(samples);
            var pipeline = mlContext.Transforms.Text.ProduceWordBags("BagOfWords", "Text", ngramLength: 1);
            var model = pipeline.Fit(dataView);

            // 2. Create the engine once
            var engine = mlContext.Model.CreatePredictionEngine<InputText, Output>(model);

            // 3. Run Predict for the first sentence
            var result1 = engine.Predict(new InputText() { Text= "This camera camera is good" });
            Console.WriteLine($"1st: [{string.Join(", ", result1.BagOfWords)}]");

            // 4. Run Predict for the second sentence
            var result2 = engine.Predict(new InputText() { Text = "My Name is Shiv" });
            Console.WriteLine($"2nd: [{string.Join(", ", result2.BagOfWords)}]");
        }
        public static void Lab13_TfIdf()
        {
            var mlContext = new MLContext();

            var samples = new[]
            {
            new InputText { Text = "This camera camera is good" },
            new InputText { Text = "This camera is bad" }
            };

            var dataView = mlContext.Data.LoadFromEnumerable(samples);

           
            var pipeline = mlContext.Transforms.Text.ProduceWordBags(
                "BagOfWords",
                "Text",
                ngramLength: 1,
                weighting: NgramExtractingEstimator.WeightingCriteria.TfIdf); 

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);

            var results = mlContext.Data.CreateEnumerable<Output>(transformedData, reuseRowObject: false);

            foreach (var r in results)
            {
                Console.WriteLine($"TF-IDF Vector: [{string.Join(", ", r.BagOfWords.Select(x => x.ToString("F3")))}]");
            }
        }
        public static void Lab14_Embedding()
        {
            var ml = new MLContext();

            var samples = new[]
            {
            new InputText { Text = "king" },
            new InputText { Text = "queen" },
            new InputText { Text = "camera" }
        };

            var data = ml.Data.LoadFromEnumerable(samples);

            var tokenizationPipeline = ml.Transforms.Text.TokenizeIntoWords(
                outputColumnName: "Tokens",
                inputColumnName: "Text");

            var embeddingPipeline = ml.Transforms.Text.ApplyWordEmbedding(
                outputColumnName: "Features",
                inputColumnName: "Tokens", 
                modelKind: Microsoft.ML.Transforms.Text.WordEmbeddingEstimator.PretrainedModelKind.GloVe50D
            );

            var pipeline = tokenizationPipeline.Append(embeddingPipeline);
            var model = pipeline.Fit(data);
            var transformed = model.Transform(data);

            var results = ml.Data.CreateEnumerable<TextFeatures>(transformed, false).ToList();


            for (int i = 0; i < results.Count; i++)
            {
                Console.WriteLine($"\nWord: {samples[i].Text}");
                Console.WriteLine("Vector (first 10 values):");
                Console.WriteLine(string.Join(", ", results[i].Features.Take(10)) + " ...");
            }
            var resultsList = results.ToList();

            var kingVector = resultsList[0].Features;
            var queenVector = resultsList[1].Features;
            var cameraVector = resultsList[2].Features;
            double distanceKingQueen = Common.CalculateCosineSimilarity(kingVector, queenVector);
            double distanceKingCamera = Common.CalculateCosineSimilarity(kingVector, cameraVector);

            Console.WriteLine($"\nDistance (King vs. Queen): {distanceKingQueen:F4}");
            Console.WriteLine($"Distance (King vs. Camera): {distanceKingCamera:F4}");
        }
        
    }
    //public class InputText { public string Text { get; set; } }
    //public class Output { public float[] BagOfWords { get; set; } }
}
