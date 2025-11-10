using Microsoft.ML;
using Microsoft.ML.AutoML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public static class  Week1
    {
        public static void Lab1_SimplestMLCodeSinglePrediction()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(Data.GetLinearInsuranceData()); // Data

            var pipeline = mlcontext.Transforms // f1 = Age + Salary
                                    .Concatenate("Features", "Age")
                                    .Append(
                                     mlcontext.Regression.Trainers
                                     .Ols(labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var pe = mlcontext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var prediction = pe.Predict(new InsuranceData { Age = 80 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();
        }
        public static void Lab2_SimplestMLCodeUsingTestData()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(Data.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(Data.GetTestData()); // Data

            var pipeline = mlcontext.Transforms // f1 = Age + Salary
                                    .Concatenate("Features", "Age")
                                    .Append(
                                     mlcontext.Regression.Trainers
                                     .FastForest(labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var predictions = model.Transform(testdata); // prediction

            var predictionEnumerable = mlcontext.Data.
                                            CreateEnumerable<InsurancePrediction>(predictions, reuseRowObject: false).ToList();

            foreach ( var prediction in predictionEnumerable)
            {
                Console.WriteLine(prediction.PredictedPremium);
            }
            
            Console.Read();
        }
        public static void Lab3and4_SimplestMLCodeCheckingRSandRMSE()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(Data.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(Data.GetTestData()); // Data

            var pipeline = mlcontext.Transforms // f1 = Age + Salary
                                    .Concatenate("Features", "Age")
                                    .Append(
                                     mlcontext.Regression.Trainers
                                     .FastForest(labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var predictions = model.Transform(testdata); // prediction

            var metrics = mlcontext.Regression.Evaluate(predictions, labelColumnName: "Premium", scoreColumnName: "Score");

            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");

            Console.Read();
        }
        public static void Lab5_SimplestMLAutoMl()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(Data.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(Data.GetTestData()); // Data

            var experimentSettings = new RegressionExperimentSettings
            {
                MaxExperimentTimeInSeconds = 30 // try every model for x sec
            };
            var experiment = mlcontext.Auto().CreateRegressionExperiment(experimentSettings);
            var result = experiment.Execute(data, labelColumnName: "Premium");
            foreach (var run in result.RunDetails)
            {
                Console.WriteLine($"Model: {run.TrainerName}");
                Console.WriteLine($"R²: {run.ValidationMetrics.RSquared}");
                Console.WriteLine($"RMSE: {run.ValidationMetrics.RootMeanSquaredError}");
                Console.WriteLine("------------------------------------");
            }
            var bestModel = result.BestRun.Model;
            Console.WriteLine($"Best Model: {result.BestRun.TrainerName}");
            Console.Read();
        }
        public static void Lab6_SimplestMLAutoMlWithHugeData()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<InsuranceData>(
            path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\linear_insurance_100k.csv",   // your CSV file path
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
        public static void Lab7_LargeFileTestingwithAutoMLOutput()
        {
            var mlContext = new MLContext();
            var data = mlContext.Data.LoadFromTextFile<InsuranceData>(
            path: "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\linear_insurance_100k.csv",   // your CSV file path
            hasHeader: true,
            separatorChar: ',');
           
            var pipeline = mlContext.Transforms // f1 = Age + Salary
                                     .Concatenate("Features", "Age")
                                     .Append(
                                      mlContext.Regression.Trainers
                                      .Sdca(labelColumnName: "Premium",
                                             featureColumnName: "Features"
                                       ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var pe = mlContext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var prediction = pe.Predict(new InsuranceData { Age = 45 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();

        }
    }
}
