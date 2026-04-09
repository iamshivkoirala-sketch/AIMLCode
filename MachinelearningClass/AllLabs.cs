using AllMiniLmL6V2Sharp;
using MachinelearningClass.Cohort;
using MachinelearningClass.InterviewQuestions;
using MachinelearningClass.ModelNLP;
using MachinelearningClass.Regression;
using Microsoft.Data.SqlClient;
using Microsoft.Data.SqlTypes;
using Microsoft.Data;
using Microsoft.ML;
using Microsoft.ML.AutoML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using OpenAI.Chat;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using AllMiniLmL6V2Sharp.Tokenizer;
using MachinelearningClass.DataNLP;
using OpenAI.Embeddings;
using OpenAI;
using System.ClientModel;

namespace MachinelearningClass
{
    public static class  AllLabs
    {
        
        public static void Lab1_SimplestMLCode()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
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
            var prediction = pe.Predict(new InsuranceData { Age = 82 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();
        }
        public static void Lab2_ModelisMathsFormula()
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
            float intercept = modelParams.Bias; 
            var slope = modelParams.Weights[0];

            Console.WriteLine($"--- Model Parameters ---");
            Console.WriteLine($"Intercept (Bias): {intercept}");
            Console.WriteLine($"Slope (Weight):   {slope}");
            Console.WriteLine($"Equation: y = {slope}x + {intercept}");
            Console.Read();
        }
        public static void Lab3_WithMultiFeature()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceDataMultiFeature()); // Data
            var pipeline = mlcontext.Transforms // f1 = Age + Salary
                                    .Concatenate("Features", "Age","HighBp","LowBp")
                                    .Append(
                                     mlcontext.Regression.Trainers.Sdca
                                     (labelColumnName: "Premium",
                                            featureColumnName: "Features"
                                      ));
            var model = pipeline.Fit(data); // execution = data + Ols ==> Model
            var pe = mlcontext.Model.
                        CreatePredictionEngine<InsuranceData, InsurancePrediction>(model);
            var prediction = pe.Predict(new InsuranceData { Age = 82 , HighBp=145 , LowBp=92 });

            Console.WriteLine(prediction.PredictedPremium);
            Console.Read();
        }
        public static void Lab4_SimplestMLCodeUsingTestData()
        {
            var mlcontext = new MLContext();
            var data = mlcontext.Data.LoadFromEnumerable(DataRegression.GetLinearInsuranceData()); // Data
            var testdata = mlcontext.Data.LoadFromEnumerable(DataRegression.GetTestData()); // Data

            var pipeline = mlcontext.Transforms 
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
        public static void Lab5_CheckingRSandRMSE()
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
        public static void Lab6_SimplestMLAutoMl()
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
                        .Append(ml.BinaryClassification.Trainers.SdcaLogisticRegression(
                        labelColumnName: "IsApple",
                        featureColumnName: "Features"));
            var model = pipeline.Fit(data);
            var engine = ml.Model.CreatePredictionEngine<Cohort.FruitData, FruitPrediction>(model);

            var test = new Cohort.FruitData { Weight = 12 };
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

            var test = new CustomerData { Age = 55, Spending = 35000 };

            var result = engine.Predict(test);

            Console.WriteLine($"Cluster: {result.PredictedClusterId}");
        }
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

            var encoded = ml.Data.CreateEnumerable<FruitFeatures>(
                transformedData, reuseRowObject: false);

            Console.WriteLine("One-Hot Encoded Vectors:");
            foreach (var row in encoded)
            {
                Console.WriteLine($"[{string.Join(",", row.FruitEncoded)}]");
            }
        }
        public static void Lab13_Bow()
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
            var result1 = engine.Predict(new InputText() { Text = "This camera camera is good" });
            Console.WriteLine($"1st: [{string.Join(", ", result1.BagOfWords)}]");

            // 4. Run Predict for the second sentence
            var result2 = engine.Predict(new InputText() { Text = "My Name is Shiv" });
            Console.WriteLine($"2nd: [{string.Join(", ", result2.BagOfWords)}]");
        }
        public static void Lab14_TfIdf()
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
        public static void Lab15_Embedding()
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
            var messages = new List<ChatMessage>();

            var key = Environment.GetEnvironmentVariable("aikey");
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            //messages.Add(new SystemChatMessage("Show him 10 C# and ASP.NET interview question list "));
            messages.Add(new SystemChatMessage(
                $"Ask only: {question}. " +
                "Display atleast 10 questions"));
            //messages.Add(new UserChatMessage(candidateExp));


            var completion = await chat.CompleteChatAsync(messages);
            string questionfromchatgpt = completion.Value.Content.Last().Text;
            Console.WriteLine($"{questionfromchatgpt}");





        }
        //The vocab.txt is used by the tokenizer to convert text into tokens (numbers),
        //and the ONNX model all-MiniLM-L6-v2.onnx is used by the embedder to
        //convert those tokens into vector embeddings that capture meaning.
        // https://huggingface.co/onnx-models/all-MiniLM-L6-v2-onnx/tree/main
        // Data folder does not have the All mini & GPT model and vocab json please download
        // from huggingface

        public static void Lab16_SimpleBertEncoding()
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
        public static void Lab17_FailedGPTEncoding()
        {
            var ml = new MLContext();
            string modelPath = Program.datapath + @"\\GPT\\model.onnx";

            // Create ONNX pipeline
            var pipeline = ml.Transforms.ApplyOnnxModel(
                outputColumnNames: new[] { "logits" },
                inputColumnNames: new[] { "input_ids" },
                modelFile: modelPath
            );

            // Create dummy input for Fit
            var dummyInput = new GPT2Input
            {
                input_ids = new long[1, 16] // all zeros
            };

            var model = pipeline.Fit(ml.Data.LoadFromEnumerable(new List<GPT2Input> { dummyInput }));

            var engine = ml.Model.CreatePredictionEngine<GPT2Input, GPT2Output>(model);

            // Example input: "I love" -> token IDs padded to length 16
            long[] tokens = { 40, 18435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
            var input = new GPT2Input
            {
                input_ids = new long[1, 16]
            };
            for (int i = 0; i < tokens.Length; i++)
                input.input_ids[0, i] = tokens[i];

            var result = engine.Predict(input);

            // Find the next token with the highest probability for last position
            int seqLen = result.logits.GetLength(1);
            int vocabSize = result.logits.GetLength(2);

            float max = float.MinValue;
            int predictedIndex = -1;

            for (int v = 0; v < vocabSize; v++)
            {
                float val = result.logits[0, seqLen - 1, v]; // last token position
                if (val > max)
                {
                    max = val;
                    predictedIndex = v;
                }
            }

            Console.WriteLine($"Predicted next token ID: {predictedIndex}");
        }

        public static async Task Lab18_SimpleChatGPTOnline()
        {
            var credential = new ApiKeyCredential(Environment.GetEnvironmentVariable("aikey"));
            var chatClient = new ChatClient("gpt-3.5-turbo", credential); // Use a standard model for generation

            string simpleSentencePrompt = "I love ";

            ChatCompletion completion = await chatClient.CompleteChatAsync(
                                        messages: new[]
                                        {
                                        new UserChatMessage(simpleSentencePrompt)
                                        }
                                        );

            string prediction = completion.Content.Last().Text;
            Console.WriteLine(prediction);
        }
        public static async Task Lab19_RAGChatGPTOnline()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            var embeddingClient = new EmbeddingClient("text-embedding-3-small", key);

            List<RAGLookup> lookupStore = DataforNlp.getRAGData();

            foreach (var item in lookupStore)
            {
                var embed = await embeddingClient.GenerateEmbeddingAsync(item.Description);
                item.DescriptionEmbedding = embed.Value.ToFloats().ToArray();

            }
            Console.WriteLine("Enter who you are ?");

            string candidateExp = Console.ReadLine();
            var candidateEmbed = await embeddingClient.GenerateEmbeddingAsync(candidateExp);
            var candidateVector = candidateEmbed.Value.ToFloats().ToArray();

            var bestMatch = lookupStore
                .OrderByDescending(x => Common.CalculateCosineSimilarity(x.DescriptionEmbedding, candidateVector))
                .First();



            // Start interview with selected questions
            var messages = new List<ChatMessage>();


            //messages.Add(new SystemChatMessage("Show him 10 C# and ASP.NET interview question list "));
            messages.Add(new SystemChatMessage(
                $"Ask only: {bestMatch.QuestionstobeAsked}. " +
                "Display atleast 10 questions"));
            //messages.Add(new UserChatMessage(candidateExp));


            var completion = await chat.CompleteChatAsync(messages);
            string questionfromchatgpt = completion.Value.Content.Last().Text;
            Console.WriteLine($"{questionfromchatgpt}");





        }
        public static async Task Lab20_PromptUnderstanding()
        {
            var key = Environment.GetEnvironmentVariable("aikey");
            var client = new OpenAIClient(key);
            var chat = new ChatClient(model: "gpt-4o-mini", key);
            var messages = new List<ChatMessage>
            {
                new SystemChatMessage("Take c# interview . Only ask ASP.NET core question if he does not answr that ask him basic OOP. Do not repeat question once asked. Ask one question at a time. Do not answer yourself.")
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
        public class Output { public float[] BagOfWords { get; set; } }
    }
}
