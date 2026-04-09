using AllMiniLmL6V2Sharp;
using AllMiniLmL6V2Sharp.Tokenizer;
using Microsoft.Data;
using Microsoft.Data.SqlClient;
using Microsoft.Data.SqlTypes;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.Embeddings;
using OpenAI;
using System.Data;
using Microsoft.Extensions.AI;
using Microsoft.SemanticKernel.Embeddings;
using Microsoft.SemanticKernel.ChatCompletion;

namespace GenericEmbeddingClient
{
    internal class Program
    {
        public static string datapath = "C:\\Users\\shivB\\source\\repos\\MachinelearningClass\\MachinelearningClass\\Data\\";

        static void Main(string[] args)
        {
            //IEmbeddingGenerator<string, Embedding<float>> generator = new 
            //        OpenAIClient("key").GetEmbeddingClient("text-embedding-3-small");
            //var client = new OpenAIClient("your-api-key");

            // 2. Wrap it in the universal interface
            // This specifies the model (e.g., "text-embedding-3-small")
            //IEmbeddingGenerator<string, Embedding<float>> generator = client.AsIEmbeddingGenerator("text-embedding-3-small");
            IEmbeddingGenerator<string, Embedding<float>> embedding = new LocalOnnxGenerator(
            Program.datapath + @"\Model.onnx",
            Program.datapath + @"\vocab.txt");
            //MakeEmbedding(embedding);
            var res = SemanticSearch(embedding).Result.ToString();
            var key = Environment.GetEnvironmentVariable("aikey");
            IChatClient chat = new OpenAI.Chat.ChatClient("gpt-4o-mini", key).AsIChatClient();
            GeneralAIChat(res,chat).Wait();
            
            Console.WriteLine("Hello, World!");
        }
        public static void MakeEmbedding(IEmbeddingGenerator<string, Embedding<float>> generator)
        {
            string connectionString =
                "Server=DESKTOP-ILFSBH1\\SQLEXPRESS01;Database=InterviewQuestions;Trusted_Connection=True;MultipleActiveResultSets=True;TrustServerCertificate=True;";

           

            using (SqlConnection con = new SqlConnection(connectionString))
            {
                con.Open();

                // Read only what you need
                using (SqlCommand selectCmd = new SqlCommand("SELECT Id, Expierience FROM dbo.tblExp1", con))
                using (SqlDataReader reader = selectCmd.ExecuteReader())
                {
                    while (reader.Read())
                    {
                        int id = reader.GetInt32(reader.GetOrdinal("Id"));
                        string experience = reader["Expierience"]?.ToString() ?? "";

                        // Generate embedding (float[])
                        //float[] embedding = embedder.GenerateEmbedding(experience).ToArray();
                        var res = generator.GenerateAsync(experience);
                        var vectorArray = res.Result.Vector;

                        var vec = new SqlVector<float>(vectorArray);

                        // Update the row by Id
                        using (SqlCommand updateCmd = new SqlCommand(
                            "UPDATE dbo.tblExp1 SET ExpVector = @vec WHERE Id = @id", con))
                        {
                            updateCmd.Parameters.Add("@id", SqlDbType.Int).Value = id;
                            updateCmd.Parameters.Add("@vec", SqlDbTypeExtensions.Vector).Value = vec;

                            updateCmd.ExecuteNonQuery();
                        }
                    }
                }
            }
        }
        public static async Task<string> SemanticSearch(
        IEmbeddingGenerator<string, Embedding<float>> embedding)
        {
            Console.Write("Enter text: ");
            var inputText = Console.ReadLine() ?? "";

            var response = await embedding.GenerateAsync([inputText]);
            float[] queryEmbedding = response.First().Vector.ToArray();

            string connectionString =
                "Server=DESKTOP-ILFSBH1\\SQLEXPRESS01;Database=InterviewQuestions;Trusted_Connection=True;MultipleActiveResultSets=True;TrustServerCertificate=True;";

            using (SqlConnection con = new SqlConnection(connectionString))
            {
                await con.OpenAsync();

                using (SqlCommand cmd = new SqlCommand("dbo.SearchExpVector", con))
                {
                    cmd.CommandType = CommandType.StoredProcedure;

                    // Native VECTOR parameter
                    cmd.Parameters.Add(new SqlParameter("@QueryVector", SqlDbTypeExtensions.Vector)
                    {
                        Value = new SqlVector<float>(queryEmbedding)
                    });

                    using (SqlDataReader reader = await cmd.ExecuteReaderAsync())
                    {
                        if (await reader.ReadAsync())
                        {
                            string experience = reader["Expierience"]?.ToString() ?? "";
                            string question = reader["Questions"]?.ToString() ?? "";
                            float distance = Convert.ToSingle(reader["Distance"]);

                            Console.WriteLine($"\nBest Match:\n{experience} ask him {question}");
                            Console.WriteLine($"Distance: {distance}");
                            return question;
                        }
                    }
                }
            }
            return "";
        }

        public static async Task GeneralAIChat(string question , IChatClient chat)
        {



            // Start interview with selected questions
            var messages = new List<ChatMessage>();

           
    
            //var chat = new ChatClient(model: "gpt-4o-mini", key);
            //messages.Add(new SystemChatMessage("Show him 10 C# and ASP.NET interview question list "));
            messages.Add(new ChatMessage(ChatRole.System,
                $"Ask only: {question}. " +
                "Display atleast 10 questions"));
            //messages.Add(new UserChatMessage(candidateExp));


            var completion = await chat.GetResponseAsync(messages);
            string questionfromchatgpt = completion.Messages.Last().Text;
            Console.WriteLine($"{questionfromchatgpt}");





        }
    }

    public class LocalOnnxGenerator : IEmbeddingGenerator<string, Embedding<float>>
    {
        private readonly AllMiniLmL6V2Embedder _embedder;
        private readonly BertTokenizer _tokenizer;

        public LocalOnnxGenerator(string modelPath, string vocabPath)
        {
            _tokenizer = new BertTokenizer(vocabPath);
            _embedder = new AllMiniLmL6V2Embedder(modelPath, _tokenizer);
        }

        public async Task<GeneratedEmbeddings<Embedding<float>>> GenerateAsync(
            IEnumerable<string> values, EmbeddingGenerationOptions? options = null, CancellationToken cancellationToken = default)
        {
            var embeddings = values.Select(text =>
                new Embedding<float>(_embedder.GenerateEmbedding(text).ToArray())
            ).ToList();

            return new GeneratedEmbeddings<Embedding<float>>(embeddings);
        }

        // Required by interface for clean-up
        public void Dispose() { /* Add cleanup if your embedder supports it */ }
        public object? GetService(Type serviceType, object? serviceKey = null) => serviceType.IsInstanceOfType(this) ? this : null;
        public EmbeddingGeneratorMetadata Metadata => new("LocalOnnx", null, "AllMiniLmL6");
    }

}
