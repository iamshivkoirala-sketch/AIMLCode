using Microsoft.ML.AutoML;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Trainers;
using Tensorflow.Keras.Engine;
using MachinelearningClass.Cohort;
using MachinelearningClass.Regression;

namespace MachinelearningClass
{
    public class Week2
    {
        public static void Lab6_SimplestMLAutoMlWithHugeData()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<InsuranceData>(
            path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\linear_insurance_100k.csv",   // your CSV file path
            hasHeader: true,
            separatorChar: ',');
            var splitData = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
            var trainData = splitData.TrainSet;
            var testData = splitData.TestSet;
            var experimentSettings = new RegressionExperimentSettings
            {
                
                MaxExperimentTimeInSeconds = 30 // try every nodel for x sec
            };
            var experiment = mlContext.Auto().CreateRegressionExperiment(experimentSettings);
            var result = experiment.Execute(data, labelColumnName: "Premium");
            foreach (var run in result.RunDetails)
            {
                Console.WriteLine($"Model: {run.TrainerName}");
                Console.WriteLine($"R²: {run.ValidationMetrics.RSquared}");
                Console.WriteLine($"RMSE: {run.ValidationMetrics.RootMeanSquaredError}");
                Console.WriteLine("------------------------------------");
            }
            // Get best model
            var bestModel = result.BestRun.Model;
            Console.WriteLine($"Best Model: {result.BestRun.TrainerName}");

        }
        public static void Lab6_LargeFileTestingwithAutoMLOutput()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<InsuranceData>(
            path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\data\\linear_insurance_100k.csv",   // your CSV file path
            hasHeader: true,
            separatorChar: ',');

            var pipeline = mlContext.Transforms // f1 = Age + Salary
                                     .Concatenate("Features", "Age")
                                     .Append(
                                      mlContext.Regression.Trainers
                                      .LightGbm(labelColumnName: "Premium",
                                             featureColumnName: "Features"
                                       ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var pe = mlContext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var prediction = pe.Predict(new InsuranceData { Age = 45 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();

        }
        public static void Lab7_SavingModel()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<InsuranceData>(
            path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\data\\linear_insurance_100k.csv",   // your CSV file path
            hasHeader: true,
            separatorChar: ',');

            var pipeline = mlContext.Transforms // f1 = Age + Salary
                                     .Concatenate("Features", "Age")
                                     .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                                     .Append(
                                      mlContext.Regression.Trainers
                                      .OnlineGradientDescent(labelColumnName: "Premium",
                                             featureColumnName: "Features"
                                       ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
           
            mlContext.Model.Save(model, data.Schema, "insuranceModel.zip");

            var pe = mlContext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var prediction = pe.Predict(new InsuranceData { Age = 45 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();

        }
       

        public static void Lab7_LoadingModel()
        {
            var mlContext = new MLContext();

            // LOAD OLD MODEL
            DataViewSchema inputSchema;
            var loadedModel = mlContext.Model.Load("insuranceModel.zip", out inputSchema);

            // NEW TRAINING DATA (new rows)
            var newData = new List<InsuranceData>
            {
            new InsuranceData { Age = 120, Premium = 70000 },
            };

            var newDataView = mlContext.Data.LoadFromEnumerable(newData);

            // RETRAIN (INCREMENTAL FIT)
            var trainer = mlContext.Regression.Trainers
                        .OnlineGradientDescent(labelColumnName: "Premium", featureColumnName: "Features")
                      
                        ;

            var modelChain = (Microsoft.ML.Data.TransformerChain<ITransformer>)loadedModel;
            IDataView preppedNewDataView = loadedModel.Transform(newDataView);

            // 2. Get the last transformer in the chain, which is the actual trained predictor.
            ITransformer finalPredictor = modelChain.Last();

            // 3. Cast the final predictor to the specific interface that holds the 'Model' property.
            // We assume object as the output type for safety, it varies by scenario.
            var singleFeaturePredictor = (ISingleFeaturePredictionTransformer<object>)finalPredictor;

            // 4. Finally, access the specific Model Parameters type.
            LinearRegressionModelParameters originalModelParameters =
                singleFeaturePredictor.Model as LinearRegressionModelParameters;

            var model2 = trainer.Fit(preppedNewDataView, originalModelParameters);
            var pe = mlContext.Model.
                       CreatePredictionEngine<InsuranceData, InsurancePrediction>(model2);
            var prediction = pe.Predict(new InsuranceData { Age = 120 });

            Console.WriteLine(prediction.PredictedPremium);


            Console.WriteLine("Model updated!");


        }
        public static void Lab8_LogisticCalssification()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetFruitData());

            var pipeline =
                        ml.Transforms.Concatenate("Features", "Weight")
                        .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
                        labelColumnName: "IsApple",
                        featureColumnName: "Features"));
            var model = pipeline.Fit(data);
            var engine = ml.Model.CreatePredictionEngine<FruitData, FruitPrediction>(model);

            var test = new FruitData { Weight = 12 };
            var result = engine.Predict(test);

            Console.WriteLine(result.PredictedLabel );
        }
        public static void Lab9_MulticlassCalssification()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetFruitData());

            var pipeline =
                ml.Transforms.Conversion.MapValueToKey("Label", nameof(FruitData.FruitType))
                .Append(ml.Transforms.Categorical.OneHotEncoding("ColorEncoded", nameof(FruitData.Color)))
                .Append(ml.Transforms.Concatenate("Features", "Weight", "ColorEncoded"))
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);
            var engine = ml.Model.CreatePredictionEngine<FruitData, FruitPrediction>(model);

            var test = new FruitData
            {
                Weight = 110,
                Color = "Yellow",
               
            };

            var result = engine.Predict(test);

            Console.WriteLine($"Predicted Type: {result.PredictedLabel}");

        }
        public static void Lab10_SimpleCustering()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetCustomerData());

            var pipeline = ml.Transforms.Concatenate("Features", "Age", "Spending")
                .Append(ml.Clustering.Trainers.KMeans(numberOfClusters: 3));

            var model = pipeline.Fit(data);

            var engine = ml.Model.CreatePredictionEngine<CustomerData, CustomerCluster>(model);

            var test = new CustomerData { Age = 55, Spending = 35000 };

            var result = engine.Predict(test);

            Console.WriteLine($"Cluster: {result.PredictedClusterId}");
        }
        

        
       


    }
}
