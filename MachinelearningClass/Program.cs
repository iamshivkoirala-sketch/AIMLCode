using MachinelearningClass.Cohort;
using MachinelearningClass.Regression;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
namespace MachinelearningClass
{
    internal class Program
    {
        public static string sqlConnectionString = "Data Source=DESKTOP-ILFSBH1\\SQLEXPRESS01;Initial Catalog=InterviewQuestions;Integrated Security=True;Trust Server Certificate=True";

        public static string datapath = "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\";
         static  void Main(string[] args)
        {
            //OtherLabs.Lab29SemanticKernel().Wait();
            AllLabs.Lab19ToolingusingSemanticKernel().Wait();
            //AllLabs.Lab21_QdrantRag().Wait();
            //ChunkingLessons.FixedAndRecursiveChunking();
            Console.Read();
        }
        
        public static void Clustering()
        {
            var ml = new MLContext();

            var data = ml.Data.LoadFromEnumerable(DataRegression.GetCustomerData());
            var pipeline = ml.Transforms.Concatenate("Features", "Age", "Spending")
                .Append(ml.Clustering.Trainers.KMeans(numberOfClusters: 3));

            var model = pipeline.Fit(data);

            var engine = ml.Model.CreatePredictionEngine<CustomerData, CustomerCluster>(model);

            var test = new CustomerData { Age = 21, Spending = 22000 };

            var result = engine.Predict(test);

            Console.WriteLine($"Cluster: {result.PredictedClusterId}");
        }
    
        public static void ReadModel()
        {
            var mlContext = new MLContext();
            DataViewSchema inputSchema;
            var loadedModel = mlContext.Model.Load("d:\\insuranceModel1.zip", out inputSchema);
            var pe = mlContext.Model.
                       CreatePredictionEngine<InsuranceData, InsurancePrediction>(loadedModel);
            var prediction = pe.Predict(new InsuranceData { Age = 63 });
            Console.WriteLine(prediction.PredictedPremium);
        }
        public static void SaveModel()
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

            mlContext.Model.Save(model, data.Schema, "d:\\insuranceModel1.zip");

        }
    }
    
}
