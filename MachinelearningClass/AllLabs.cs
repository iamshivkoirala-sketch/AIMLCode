using AllMiniLmL6V2Sharp;
using AllMiniLmL6V2Sharp.Tokenizer;
using MachinelearningClass.Cohort;
using MachinelearningClass.DataNLP;
using MachinelearningClass.InterviewQuestions;
using MachinelearningClass.ModelNLP;
using MachinelearningClass.Regression;
using Microsoft.Data;
using Microsoft.Data.SqlClient;
using Microsoft.Data.SqlTypes;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.AI;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using Microsoft.SemanticKernel.Embeddings;
using OllamaSharp;
using OpenAI;
using OpenAI.Chat;
using OpenAI.Embeddings;
using System;
using System.ClientModel;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static TorchSharp.torch.nn;
using ChatMessage = OpenAI.Chat.ChatMessage;
namespace MachinelearningClass
{
    public static class  AllLabs
    {
        // Lab 1 :- Demo using Excel.
        public static void Lab2_SimplestMLCode()
        {
            var mlcontext = new MLContext();
            
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
            var pipeline = mlcontext.Transforms
                                    
                                    // 1. Transformation
                                    .ReplaceMissingValues("Age", replacementMode: 
                                    MissingValueReplacingEstimator.ReplacementMode.Mean)
                                    .Append(mlcontext.Transforms.Concatenate("Features", "Age"))
                                    // 2. Train: Use Age (Features) to predict the Premium (Label)
                                    .Append(
                                        mlcontext.Regression.Trainers
                                        .Sdca(labelColumnName: "Premium",
                                        featureColumnName: "Features")
                                    );


            var model = pipeline.Fit(data); // Training

            var pe = mlcontext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            
            var prediction = pe.Predict(new InsuranceData { Age = 82 }); // Inference

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();
        }
        public static void Lab3_ModelisMathsFormula()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
            var pipeline = mlcontext.Transforms 
                                    .Concatenate("Features", "Age")
                                    .Append(
                                     mlcontext.Regression.Trainers
                                     .Ols(labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); 
            var pipelinesteps = model as IEnumerable<ITransformer>;

            var modelLastStep = pipelinesteps.Last() as RegressionPredictionTransformer<OlsModelParameters>;
            var modelParams = modelLastStep.Model;
            float intercept = modelParams.Bias; // intercept
            var slope = modelParams.Weights[0]; // slope

            Console.WriteLine($"--- Model Parameters ---");
            Console.WriteLine($"Intercept (Bias): {intercept}");
            Console.WriteLine($"Slope (Weight):   {slope}");
            Console.WriteLine($"Equation: y = {slope}x + {intercept}");
            Console.Read();
        }
        public static void Lab4_WithMultiFeature()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceDataMultiFeature()); // Data
            var pipeline = mlcontext.Transforms // f1 = Age + Salary
                                    .Concatenate("Features", "Age","HighBp","LowBp")
                                    .Append(
                                     mlcontext.Regression.Trainers.FastTree
                                     (labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var preview = pipeline.Fit(data).Transform(data).Preview(); // see vectors

            // training
            var pe = mlcontext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
           // inference
            var prediction = pe.Predict(new InsuranceData { Age = 82 , HighBp=145 , LowBp=92 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();
        }
        public static void Lab5_SimplestMLCodeUsingTestData()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(DataRegression.GetTestData()); // Data

            var pipeline = mlcontext.Transforms 
                                    .Concatenate("Features", "Age")
                                    .Append(
                                     mlcontext.Regression.Trainers
                                     .Ols(labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); 
            var predictions = model.Transform(testdata); 

            var predictionEnumerable = mlcontext.Data.
                                            CreateEnumerable<InsurancePrediction>(predictions, 
                                            reuseRowObject: false).ToList();
            var metrics = mlcontext.Regression.Evaluate(predictions, labelColumnName: "Premium", scoreColumnName: "Score");

            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
            foreach ( var prediction in predictionEnumerable)
            {
                Console.WriteLine(prediction.PredictedPremium- metrics.RootMeanSquaredError);
            }
            
            Console.Read();
        }
        public static void Lab6_CheckingRSandRMSE()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(DataRegression.GetTestData()); // Data

            var pipeline = mlcontext.Transforms // f1 = Age + Salary
                                    .Concatenate("Features", "Age")
                                    .Append(
                                     mlcontext.Regression.Trainers
                                     .Sdca(labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var predictions = model.Transform(testdata); // prediction

            var metrics = mlcontext.Regression.Evaluate(predictions, labelColumnName: "Premium", scoreColumnName: "Score");

            Console.WriteLine($"R-Squared: {metrics.RSquared}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");

            Console.Read();
        }
        public static void Lab7_SimplestMLAutoMl()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(DataRegression.GetTestData()); // Data

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
        public static void Lab7_SimplestMLAutoMlWithHugeData()
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
        public static void Lab7_LargeFileTestingwithAutoMLOutput()
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
        public static void Lab8_SavingModel()
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
        public static void Lab8_LoadingModel()
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
        public static void Lab9_LogisticCalssification()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetFruitData());

            var pipeline =
                        ml.Transforms.Concatenate("Features", "Weight")
                        .Append(ml.BinaryClassification.Trainers.FastTree(
                        labelColumnName: "IsApple",
                        featureColumnName: "Features"));
            var model = pipeline.Fit(data);
            var pre = pipeline.Fit(data).Preview(data);
            var engine = ml.Model.CreatePredictionEngine<Cohort.FruitData, FruitPrediction>(model);

            var test = new Cohort.FruitData { Weight = 20 };
            var result = engine.Predict(test);

            Console.WriteLine(result.PredictedLabel);
        }
        public static void Lab10_MulticlassCalssification()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetFruitData());

            var pipeline =
                ml.Transforms.Conversion.MapValueToKey("Label", nameof(Cohort.FruitData.FruitType))
                .Append(ml.Transforms.Categorical.OneHotEncoding("ColorEncoded", nameof(Cohort.FruitData.Color)))
                .Append(ml.Transforms.Concatenate("Features", "Weight", "ColorEncoded"))
                .Append(ml.MulticlassClassification.Trainers.SdcaMaximumEntropy())
                .Append(ml.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            var model = pipeline.Fit(data);
            var preview = pipeline.Fit(data).Preview(data);

            var engine = ml.Model.CreatePredictionEngine<Cohort.FruitData, FruitPrediction>(model);

            var test = new Cohort.FruitData
            {
                Weight = 110,
                Color = "Yellow",

            };

            var result = engine.Predict(test);

            Console.WriteLine($"Predicted Type: {result.PredictedLabel}");

        }
        public static void Lab11_SimpleCustering()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetCustomerData());

            var pipeline = ml.Transforms.Concatenate("Features", "Age", "Spending")
                .Append(ml.Clustering.Trainers.KMeans(numberOfClusters: 3));

            var model = pipeline.Fit(data);

            var engine = ml.Model.CreatePredictionEngine<CustomerData, CustomerCluster>(model);

            var test = new CustomerData { Age = 70, Spending = 20000 };

            var result = engine.Predict(test);

            Console.WriteLine($"Cluster: {result.PredictedClusterId}");
        }
        // Date and Numbers Time regression

        public static void Lab12_OneHotEncoding()
        {
            var ml = new MLContext();

            var data = new[]
            {
            new ModelNLP.FruitData { Fruit = "Mango" },
            new ModelNLP.FruitData { Fruit = "Apple" },
            new ModelNLP.FruitData { Fruit = "Berry" }
        };

            var dataView = ml.Data.LoadFromEnumerable(data);

            var pipeline = ml.Transforms.Categorical.OneHotEncoding(
                outputColumnName: "FruitEncoded",
                inputColumnName: "Fruit");

            var model = pipeline.Fit(dataView);
            var transformedData = model.Transform(dataView);
            // take a tex input 
            // Mango - [1,0,0]
            // [1,0,0] - Mango
            var encoded = ml.Data.CreateEnumerable<FruitFeatures>(
                transformedData, reuseRowObject: true);

            Console.WriteLine("One-Hot Encoded Vectors:");
            foreach (var row in encoded)
            {
               
                Console.WriteLine($"[{string.Join(",", row.FruitEncoded)}]");
            }
            string input = "Mango";

            var predictionEngine =
            ml.Model.CreatePredictionEngine<ModelNLP.FruitData, FruitFeatures>(
            model);

            var inputResult = predictionEngine.Predict(
            new ModelNLP.FruitData
            {
                Fruit = input
            });

            Console.WriteLine(
            $"Input vector: [{string.Join(",", inputResult.FruitEncoded)}]");
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

            // 3. Compare for the first sentence
            var result1 = engine.Predict(new InputText() { Text = "This camera camera is good" });
            Console.WriteLine($"1st: [{string.Join(", ", result1.BagOfWords)}]");

            // 4. Compare for the second sentence
            var result2 = engine.Predict(new InputText() { Text = "This camera is bad" });
            Console.WriteLine($"2nd: [{string.Join(", ", result2.BagOfWords)}]");
            // comparison vector
            //Common.CalculateEuclideanDistance()
        }
        public static void Lab12_TfIdf()
        {
            var mlContext = new MLContext();

            var samples = new[]
            {
            new InputText { Text = "The Car is Driven on the Highway." },
            new InputText { Text = "The Truck is Driven on the Road." }
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
            // home extract the prediction engine
            // Car is on Highway
            // Cosine
        }
        public static void Lab13_Embedding()
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
            double similiarityKingQueen = Common.CalculateCosineSimilarity(kingVector, queenVector);
            double similiarityCamera = Common.CalculateCosineSimilarity(kingVector, cameraVector);

            Console.WriteLine($"\nDistance (King vs. Queen): {similiarityKingQueen:F4}");
            Console.WriteLine($"Distance (King vs. Camera): {similiarityCamera:F4}");
        }

      
        //The vocab.txt is used by the tokenizer to convert text into tokens (numbers),
        //and the ONNX model all-MiniLM-L6-v2.onnx is used by the embedder to
        //convert those tokens into vector embeddings that capture meaning.
        // https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/tree/main
        // Data folder does not have the All mini & GPT model and vocab json please download
        // from huggingface

        public static void Lab14_SimpleBertEncoding()
        {
            var tokenizer = new BertTokenizer(Program.datapath + @"\\vocab.txt");
            var embedder = new AllMiniLmL6V2Embedder(

                Program.datapath + @"\\Model.onnx",
                tokenizer
            );
            string texttobeMatched = "I love cricket and especially batting.";
            var texttobeMatchedV = embedder.GenerateEmbedding(texttobeMatched).ToArray();

            Console.WriteLine("Enter text be matched");
            string inputText = Console.ReadLine();
            var inputTextV = embedder.GenerateEmbedding(inputText).ToArray();
            Console.WriteLine(Common.CalculateCosineSimilarity(texttobeMatchedV, inputTextV));


        }

        public static async Task Lab17_OpenAILLMEmbeddings()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);

            List<RAGLookup> lookupStore = DataforNlp.getRAGData();

            foreach (var item in lookupStore)
            {
                var embed = await embeddingClient.GenerateEmbeddingAsync(item.Description);
                item.DescriptionEmbedding = embed.Value.ToFloats().ToArray();

            }
            while (1 == 1)
            {
                Console.WriteLine("Enter who you are ?");

                string candidateExp = Console.ReadLine();
                var candidateEmbed = await embeddingClient.GenerateEmbeddingAsync(candidateExp);
                var candidateVector = candidateEmbed.Value.ToFloats().ToArray();

                var bestMatch = lookupStore
                    .OrderByDescending(x => Common.CalculateCosineSimilarity(x.DescriptionEmbedding, candidateVector))
                    .First();

                Console.WriteLine(bestMatch.Description);
            }



        }
        public static async Task Lab18_OpenAIExamplewithPromptBasics()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var chat = new ChatClient(model: "gpt-4o-mini", key);


            Console.WriteLine("Enter who you are ?");

            string candidateExp = Console.ReadLine();
            var messages = new List<ChatMessage>();


            //messages.Add(new SystemChatMessage("Show him 10 C# and ASP.NET interview question list "));
            messages.Add(new SystemChatMessage(
                "As per " + candidateExp + "   display one question after other C# , ASP.NET , SQL Server and check answers for all questions at last. " +
                "Questions should not repeat"));

            while (1 == 1)
            {
                var completion = await chat.CompleteChatAsync(messages);
                string questionfromchatgpt = completion.Value.Content.Last().Text;
                messages.Add(new AssistantChatMessage(questionfromchatgpt));

                Console.WriteLine($"{questionfromchatgpt}");
                Console.WriteLine("ENter Answer");
                string answer = Console.ReadLine();
                messages.Add(new UserChatMessage(answer));



                ChatTokenUsage usage = completion.Value.Usage;

                Console.WriteLine("\n--- Token Consumption Report ---");
                Console.WriteLine($"Input (Prompt) Tokens: {usage.InputTokenCount}");
                Console.WriteLine($"Output (Completion) Tokens: {usage.OutputTokenCount}");
                Console.WriteLine($"Total Tokens Consumed: {usage.TotalTokenCount}");
            }
        }
        public static async Task Lab18_OpenAIExamplewithPromptBasics2Improved()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var chat = new ChatClient(model: "gpt-4o-mini", key);

            Console.WriteLine("Enter who you are ?");
            string candidateExp = Console.ReadLine();

            var messages = new List<ChatMessage>
    {
        new SystemChatMessage(
            $"You are a technical interviewer. Based on this experience: '{candidateExp}', " +
            "generate exactly 10 distinct, non-overlapping interview questions covering C#, ASP.NET, and SQL Server. " +
            "Return ONLY the questions, each on a fresh new line starting with its number (e.g., '1. ', '2. '). " +
            "Do NOT include introductory remarks, markdown formatting, greetings, or conversational filler text.")
    };

            Console.WriteLine("Open AI Call...");

            var completion = await chat.CompleteChatAsync(messages);
            string fullResponse = completion.Value.Content.Last().Text;

            ChatTokenUsage usage = completion.Value.Usage;
            Console.WriteLine($"Input (Prompt) Tokens: {usage.InputTokenCount}");
            Console.WriteLine($"Output (Completion) Tokens: {usage.OutputTokenCount}");
            Console.WriteLine($"Total Tokens Consumed: {usage.TotalTokenCount}");

            string[] interviewQuestions = fullResponse.Split(new[] { "\n", "\r\n" }, StringSplitOptions.RemoveEmptyEntries);

            

            string[] candidateAnswers = new string[interviewQuestions.Length];

            for (int i = 0; i < interviewQuestions.Length; i++)
            {
                Console.WriteLine($"\n{interviewQuestions[i]}");
                Console.WriteLine("Enter Answer:");

                candidateAnswers[i] = Console.ReadLine();
            }

            Console.WriteLine("Interview Complete!");
            Console.WriteLine($"Successfully recorded responses for all {interviewQuestions.Length} questions locally.");
        }
        public static async Task<List<RAGLookup>> BuildInmemoryRag()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            List<RAGLookup> lookupStore = DataforNlp.getRAGData();

            foreach (var item in lookupStore)
            {
                var embed = await embeddingClient.GenerateEmbeddingAsync(item.Description);
                item.DescriptionEmbedding = embed.Value.ToFloats().ToArray();
                // take all question and store in DB for now DB is inmemory
                var messages = new List<ChatMessage>
                {
                new SystemChatMessage(
                $"You are a technical interviewer. Based on this experience: '{item.Description} ' , " +
                "generate exactly 10 distinct, non-overlapping interview questions covering + " + item.QuestionstobeAsked  + 
                ".Return ONLY the questions, each on a fresh new line starting with its number (e.g., '1. ', '2. '). " +
                "Do NOT include introductory remarks, markdown formatting, greetings, or conversational filler text.")
                };

                Console.WriteLine("Open AI Calling and loading questions for " + item.Description);

                var completion = await chat.CompleteChatAsync(messages);
                string fullResponse = completion.Value.Content.Last().Text;

                ChatTokenUsage usage = completion.Value.Usage;
                Console.WriteLine($"Input (Prompt) Tokens: {usage.InputTokenCount}");
                Console.WriteLine($"Output (Completion) Tokens: {usage.OutputTokenCount}");
                Console.WriteLine($"Total Tokens Consumed: {usage.TotalTokenCount}");

                string[] interviewQuestions = fullResponse.Split(new[] { "\n", "\r\n" }, StringSplitOptions.RemoveEmptyEntries);

                foreach (var question in interviewQuestions)
                {
                    item.Questions.Add(question);
                }
               

            }
            return lookupStore;
        }
        
        public static async Task Lab19_Rag()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            List<RAGLookup> lookupStore = await BuildInmemoryRag();
            while (1 == 1)
            {
                Console.WriteLine("Enter who you are ?");

                string candidateExp = Console.ReadLine();
                var candidateEmbed = await embeddingClient.GenerateEmbeddingAsync(candidateExp);
                var candidateVector = candidateEmbed.Value.ToFloats().ToArray();

                var bestMatch = lookupStore
                    .OrderByDescending(x => Common.CalculateCosineSimilarity(x.DescriptionEmbedding, candidateVector))
                    .First();

                Console.WriteLine(bestMatch.Description);
                foreach (var q in bestMatch.Questions)
                {
                    Console.WriteLine(q);
                }
            }
        }
        public static async Task Lab20_RagWithSQLServer()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            while (1 == 1)
            {
                Console.WriteLine("Enter who you are ?");

                string candidateExp = Console.ReadLine();
                var candidateEmbed = await embeddingClient.GenerateEmbeddingAsync(candidateExp);
                var candidateVector = candidateEmbed.Value.ToFloats().ToArray();

                var bestMatch = GetBestTopMatch(candidateVector);

                Console.WriteLine(bestMatch.Description);
                foreach (var q in bestMatch.Questions)
                {
                    Console.WriteLine(q);
                }
            }
        }
        public static async Task Lab19_PromptUnderstanding()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var client = new OpenAIClient(key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            var messages = new List<ChatMessage>
            {
                new SystemChatMessage("Take c# interview . Only ask ASP.NET core question if he does not answr " +
                "                   that ask him basic OOP. Do not repeat question once asked. Ask one question at a time. " +
                "                   Do not answer yourself.")
            };
            while (true)
            {
                var completion = await chat.CompleteChatAsync(messages);
                string questionfromchatgpt = completion.Value.Content.Last().Text;
                messages.Add(new AssistantChatMessage(questionfromchatgpt));
                Console.WriteLine($"{questionfromchatgpt}");
                var userResponse = "";
                userResponse = Console.ReadLine(); // answr
                messages.Add(new UserChatMessage(userResponse)); // chat gpt

            }


        }
        public static async Task ProjectPOCForIChatClient()
        {
            // -----------------------------
            // OLLAMA
            // -----------------------------
            IChatClient chatClient = new OllamaApiClient(
                new Uri("http://localhost:11434"),
                "llama3.2"
            );




            var ollamaResponse =
                await chatClient.GetResponseAsync(
                    "Explain Dependency Injection in C#.");

            Console.WriteLine("OLLAMA RESPONSE");
            Console.WriteLine(ollamaResponse.Text);

            // -----------------------------
            // OPENAI
            // -----------------------------
            var openAiClient =
                new OpenAIClient(Environment.GetEnvironmentVariable("aikey"));

            IChatClient openAiChatClient =
                openAiClient.GetChatClient("gpt-4o-mini")
                            .AsIChatClient();

            var openAiResponse =
                await openAiChatClient.GetResponseAsync(
                    "Explain Dependency Injection in C#.");

            Console.WriteLine();
            Console.WriteLine("OPENAI RESPONSE");
            Console.WriteLine(openAiResponse.Text);
            Console.ReadLine();
        }
        //public static async Task ProjectPOCEmbedding()
        //{
        //    string text = "C# ASP.NET Core Microservices";

        //    IChatClient chatClient = new OllamaApiClient(
        //       new Uri("http://localhost:11434"),
        //       "llama3.2"
        //   );

        //    IEmbeddingGenerator<string, Embedding<float>> generator =
        //    new OllamaApiClient(new Uri("http://localhost:11434"))
        //    .AsEmbeddingGenerationService(modelId: "nomic-embed-text");

        //    var embedding =
        //        await generator.GenerateAsync(
        //            "What is Dependency Injection?");

        //    Console.WriteLine(
        //        embedding.Vector.Length);

        //    // -----------------------------------
        //    // OpenAI Embeddings
        //    // -----------------------------------
        //    var openAIClient =
        //        new OpenAIClient(Environment.GetEnvironmentVariable("aikey"));

        //    IEmbeddingGenerator<string, Embedding<float>> openAIGenerator =
        //        openAIClient
        //            .GetEmbeddingClient("text-embedding-3-small")
        //            .AsIEmbeddingGenerator();

        //    Embedding<float> openAIEmbedding =
        //        await openAIGenerator.GenerateAsync(text);

        //    Console.WriteLine(
        //        $"OpenAI Vector Size : {openAIEmbedding.Vector.Length}");
        //}


        public class BertInput
        {
            [VectorType]
            public long[] input_ids { get; set; }
            [VectorType]
            public long[] attention_mask { get; set; }
        }

        public class BertOutput
        {
            [VectorType]
            public float[] sentence_embedding { get; set; }
        }

        public class QAItem
        {
            public string Question { get; set; }
            public string Answer { get; set; }
            public float[] Embedding { get; set; }
        }
        public class InputText { public string Text { get; set; } }
        public class Output {
            public string Text { get; set; }
            public float[] BagOfWords { get; set; } 
        }
        public static void BuildSQLServerRAG()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            List<RAGLookup> lookupStore = GetCurrentExperience(); // GetCurrentExperience()

            foreach (var item in lookupStore)
            {
                var embed = embeddingClient.GenerateEmbeddingAsync(item.Description).Result;

                item.DescriptionEmbedding = embed.Value.ToFloats().ToArray();
                UpdateEmbeddingFortheExperience(item);
                // take all question and store in DB for now DB is inmemory
                var messages = new List<ChatMessage>
                {
                new SystemChatMessage(
                $"You are a technical interviewer. Based on this experience: '{item.Description} ' , " +
                "generate exactly 10 distinct, non-overlapping interview questions covering + " + item.QuestionstobeAsked  +
                ".Return ONLY the questions, each on a fresh new line starting with its number (e.g., '1. ', '2. '). " +
                "Do NOT include introductory remarks, markdown formatting, greetings, or conversational filler text.")
                };

                Console.WriteLine("Open AI Calling and loading questions for " + item.Description);

                var completion = chat.CompleteChatAsync(messages).Result;
                string fullResponse = completion.Value.Content.Last().Text;

                ChatTokenUsage usage = completion.Value.Usage;
                Console.WriteLine($"Input (Prompt) Tokens: {usage.InputTokenCount}");
                Console.WriteLine($"Output (Completion) Tokens: {usage.OutputTokenCount}");
                Console.WriteLine($"Total Tokens Consumed: {usage.TotalTokenCount}");

                string[] interviewQuestions = fullResponse.Split(new[] { "\n", "\r\n" }, StringSplitOptions.RemoveEmptyEntries);

                foreach (var question in interviewQuestions)
                {
                    InsertQuestionForThatExperience(item.Id,question);
                   
                }


            }

        }
        public static List<RAGLookup> GetCurrentExperience()
        {
            var list = new List<RAGLookup>();

            using (SqlConnection con = new SqlConnection(Program.sqlConnectionString))
            {
                string query = @"
            SELECT Id, Expierience, Questions
            FROM dbo.tblExp1";

                using (SqlCommand cmd = new SqlCommand(query, con))
                {
                    con.Open();

                    using (SqlDataReader dr = cmd.ExecuteReader())
                    {
                        while (dr.Read())
                        {
                            list.Add(new RAGLookup
                            {
                                Id = Convert.ToInt32(dr["Id"]),
                                Description = dr["Expierience"].ToString(),
                                QuestionstobeAsked = dr["Questions"]?.ToString()
                            });
                        }
                    }
                }
            }

            return list;
        }
        public static void UpdateEmbeddingFortheExperience(RAGLookup item)
        {
            string vectorValue = "[" + string.Join(",", item.DescriptionEmbedding) + "]";

            using (SqlConnection con = new SqlConnection(Program.sqlConnectionString))
            {
                string query = @"
            UPDATE dbo.tblExp1
            SET ExpVector = CAST(@ExpVector AS vector(1536))
            WHERE Id = @Id";

                using (SqlCommand cmd = new SqlCommand(query, con))
                {
                    cmd.Parameters.Add("@Id", SqlDbType.Int).Value = item.Id;
                    cmd.Parameters.Add("@ExpVector", SqlDbType.NVarChar, -1).Value = vectorValue;

                    con.Open();
                    cmd.ExecuteNonQuery();
                }
            }
        }
        public static void InsertQuestionForThatExperience(int expId, string question)
        {
            using (SqlConnection con = new SqlConnection(Program.sqlConnectionString))
            {
                string query = @"
            INSERT INTO dbo.tblQuestions (ExpId, Question)
            VALUES (@ExpId, @Question)";

                using (SqlCommand cmd = new SqlCommand(query, con))
                {
                    cmd.Parameters.Add("@ExpId", SqlDbType.Int).Value = expId;
                    cmd.Parameters.Add("@Question", SqlDbType.VarChar, 1000).Value = question;

                    con.Open();
                    cmd.ExecuteNonQuery();
                }
            }
        }
        public static RAGLookup GetBestTopMatch(float[] queryVector)
        {
            string vectorValue = "[" + string.Join(",", queryVector) + "]";

            RAGLookup result = null;

            using (SqlConnection con = new SqlConnection(Program.sqlConnectionString))
            {
                con.Open();

                string sql = @"
SELECT TOP (1)
    Id,
    Expierience,
    Questions,
    VECTOR_DISTANCE(
        'cosine',
        ExpVector,
        CAST(@QueryVector AS vector(1536))
    ) AS Score
FROM dbo.tblExp1
WHERE ExpVector IS NOT NULL
ORDER BY VECTOR_DISTANCE(
        'cosine',
        ExpVector,
        CAST(@QueryVector AS vector(1536)));";

                using (SqlCommand cmd = new SqlCommand(sql, con))
                {
                    cmd.Parameters.Add("@QueryVector", SqlDbType.NVarChar, -1).Value = vectorValue;

                    using (SqlDataReader dr = cmd.ExecuteReader())
                    {
                        if (dr.Read())
                        {
                            result = new RAGLookup
                            {
                                Id = Convert.ToInt32(dr["Id"]),
                                Description = dr["Expierience"].ToString(),
                                QuestionstobeAsked = dr["Questions"]?.ToString(),
                                DescriptionEmbedding = queryVector
                            };
                        }
                    }
                }

                if (result != null)
                {
                    string questionSql = @"
SELECT Question
FROM dbo.tblQuestions
WHERE ExpId = @ExpId
ORDER BY Id;";

                    using (SqlCommand cmd = new SqlCommand(questionSql, con))
                    {
                        cmd.Parameters.Add("@ExpId", SqlDbType.Int).Value = result.Id;

                        using (SqlDataReader dr = cmd.ExecuteReader())
                        {
                            while (dr.Read())
                            {
                                result.Questions.Add(dr["Question"].ToString());
                            }
                        }
                    }
                }
            }

            return result;
        }
        public static string SemanticSearch()
        {
            string connectionString =
                "Server=DESKTOP-ILFSBH1\\SQLEXPRESS01;" +
                "Database=InterviewQuestions;" +
                "Trusted_Connection=True;" +
                "Encrypt=True;TrustServerCertificate=True;";

            var tokenizer = new BertTokenizer(Program.datapath + @"\\vocab.txt");
            var embedder = new AllMiniLmL6V2Embedder(
                Program.datapath + @"\\Model.onnx",
                tokenizer
            );

            Console.Write("Enter text: ");
            var inputText = Console.ReadLine() ?? "";

            float[] queryEmbedding = embedder.GenerateEmbedding(inputText).ToArray();

            if (queryEmbedding.Length != 384)
                throw new Exception($"Embedding length {queryEmbedding.Length}, expected 384.");

            using (SqlConnection con = new SqlConnection(connectionString))
            {
                con.Open();

                using (SqlCommand cmd = new SqlCommand("dbo.SearchExpVector", con))
                {
                    cmd.CommandType = CommandType.StoredProcedure;


                    // Native VECTOR parameter
                    cmd.Parameters.Add(new SqlParameter("@QueryVector", SqlDbTypeExtensions.Vector)
                    {
                        Value = new SqlVector<float>(queryEmbedding)
                    });

                    using (SqlDataReader reader = cmd.ExecuteReader())
                    {
                        if (reader.Read())
                        {
                            string experience = reader["Expierience"]?.ToString() ?? "";
                            string question = reader["Questions"]?.ToString() ?? "";
                            float distance = Convert.ToSingle(reader["Distance"]);

                            Console.WriteLine();
                            Console.WriteLine("Best Match:");
                            Console.WriteLine($"{experience} ask him {question}");
                            Console.WriteLine($"Distance: {distance}");
                            return question;
                        }
                        else
                        {
                            Console.WriteLine("No results found.");
                        }
                    }
                }
            }
            return "";
        }
        public static async Task ChatGptWith3Mini()
        {

            string question = SemanticSearch();


            // Start interview with selected questions
            var messages = new List<OpenAI.Chat.ChatMessage>();

            var key = Environment.GetEnvironmentVariable("aikey");
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            //messages.Add(new SystemChatMessage("Show him 10 C# and ASP.NET interview question list "));
            messages.Add(new OpenAI.Chat.SystemChatMessage(
                $"Ask only: {question}. " +
                "Display atleast 10 questions"));
            //messages.Add(new UserChatMessage(candidateExp));


            var completion = await chat.CompleteChatAsync(messages);
            string questionfromchatgpt = completion.Value.Content.Last().Text;
            Console.WriteLine($"{questionfromchatgpt}");





        }
    }
}
