using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.ServiceProcess;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Diagnostics;
using Microsoft.Data.Analysis;
using MathNet.Numerics.Statistics;
using static TorchSharp.torch.utils;

namespace MachinelearningClass
{
    public static class OtherLabs
    {
        public static void Lab23ConsumingONNX()
        {

            using var session = new InferenceSession(Program.datapath + "\\taxi_fare_model.onnx");

            float[] inputData =
            {
            1.0f,   // distance
            15.0f,  // time
            1.0f,   // traffic
            1.0f    // night
        };

            var inputTensor = new DenseTensor<float>(inputData, new[] { 1, 4 });

            var inputs = new List<NamedOnnxValue>
            {
            NamedOnnxValue.CreateFromTensor("features", inputTensor)
            };

            using var results = session.Run(inputs);

            float predictedFare = results
                .First(r => r.Name == "fare")
                .AsTensor<float>()[0];

            Console.WriteLine($"Predicted Fare: {predictedFare}");
        }
        public static async Task Lab29SemanticKernel()
        {
           
            //EXEC xp_logevent 50001,'TEST: SQL Server stopped - Agentic AI Demo','ERROR';
            var apiKey = Environment.GetEnvironmentVariable("aikey");

            var kernel = Kernel.CreateBuilder()
                .AddOpenAIChatCompletion("gpt-4o-mini", apiKey)
                .Build();

            kernel.Plugins.AddFromType<MyTools>("eventTools");

            var chat = kernel.GetRequiredService<IChatCompletionService>();

            Console.WriteLine("Agent started. Monitoring Application log for ERRORs...");

            string? lastSeenErrorSignature = null;

            while (true)
            {
                try
                {
                    var tools = new MyTools();
                    var raw = tools.GetCurrentEventViewerMessage("Application");



                    var signature = raw.GetHashCode().ToString();

                    if (signature == lastSeenErrorSignature)
                    {
                        await Task.Delay(TimeSpan.FromSeconds(30));
                        continue;
                    }

                    lastSeenErrorSignature = signature;

                    var history = new ChatHistory();
                    history.AddSystemMessage("""
                    You are an on-call monitoring agent.
                    You have access to the tool eventTools.GetCurrentEventViewerMessage.
                    Rules:
                    - If a new ERROR exists, produce:
                    - 1 line headline
                    - 2-4 bullets: likely cause + immediate checks
                    - Do not invent details.
                    - Keep it short.

                    - if you see error message SQL Server stopped try restarting it eventTools.calling RestartSQlServer
                    """);

                    history.AddUserMessage($"""
                    A new error was detected in Event Viewer (Application).
                    Summarize it and suggest immediate checks.

                    Raw event:
                    {raw}
                    """);

                    var executionSettings = new OpenAIPromptExecutionSettings
                    {
                        FunctionChoiceBehavior = FunctionChoiceBehavior.Auto()
                    };


                    ChatMessageContent response = await chat.GetChatMessageContentAsync(
                    history,// prompt
                    executionSettings,
                    kernel // LLM + tools
                    );

                    Console.WriteLine($"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}] NEW ERROR ALERT");
                    Console.WriteLine(response.Content);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Agent failure: {ex.Message}");
                }

                await Task.Delay(TimeSpan.FromSeconds(30));
            }
        }
        public static void Lab32DataQuality()
        {
            DataFrame df = DataFrame.LoadCsv(Program.datapath + "DataForQuality.csv");


            //var age1 = df["Age"].DropNulls().Cast<double>();
            var age = df["Age"].DropNulls().Cast<float>().Select(x => (double)x);
            var height = df["Height"].DropNulls().Cast<float>().Select(x => (double)x);

            //var height = df["Height"].DropNulls().Cast<double>();
            var ageStats = new DescriptiveStatistics(age);
            var heightStats = new DescriptiveStatistics(height);

            Console.WriteLine($"Remaining Rows: {df.Rows.Count}");
            Console.WriteLine($"Age Min:        {ageStats.Minimum}");
            Console.WriteLine($"Age Max:        {ageStats.Maximum}");
            Console.WriteLine($"Age mean:        {ageStats.Mean}");
            Console.WriteLine($"Age median:        {age.Median()}");

            Console.WriteLine($"Age std:        {ageStats.StandardDeviation}");
            Console.WriteLine($"Age CV:        {ageStats.StandardDeviation/ ageStats.Mean}");

            Console.WriteLine($"Age skew:        {ageStats.Skewness}");
            Console.WriteLine($"Age Kur:        {ageStats.Kurtosis}");
            Console.WriteLine("=====================================");


            Console.WriteLine($"Height Min:        {heightStats.Minimum}");
            Console.WriteLine($"Height Max:        {heightStats.Maximum}");
            Console.WriteLine($"Height Mean:        {heightStats.Mean}");
            Console.WriteLine($"Height Median:        {height.Median()}");
            Console.WriteLine($"Height standard:        {heightStats.StandardDeviation}");
            Console.WriteLine($"Height CV:        {heightStats.StandardDeviation / heightStats.Mean}");

            Console.WriteLine($"Height skew:        {heightStats.Skewness}");
            Console.WriteLine($"Height Kur:        {heightStats.Kurtosis}");

            Console.WriteLine("=====================================");
            double q1 = height.Percentile(25);
            double q3 = height.Percentile(75);
            double iqr = q3 - q1;

            // 2. Calculate the Fences
            // NARROW (1.0)
            double lowerNarrow = q1 - (1.0 * iqr);
            double upperNarrow = q3 + (1.0 * iqr);

            double lowerInner = q1 - (1.5 * iqr);
            double upperInner = q3 + (1.5 * iqr);

            double lowerOuter = q1 - (3.0 * iqr);
            double upperOuter = q3 + (3.0 * iqr);

            // 3. Print the Quality Report
          
            Console.WriteLine($"Normal Range (IQR): {q1:N1} to {q3:N1}");
            Console.WriteLine($"Inner Fence (1.5):  {lowerInner:N1} to {upperInner:N1}");
            Console.WriteLine($"Outer Fence (3.0):  {lowerOuter:N1} to {upperOuter:N1}");
            var garbageValues = height.Where(x => x < lowerOuter || x > upperOuter).ToArray();

            // 3. Get the Range of the Garbage itself
            if (garbageValues.Any())
            {
                double garbageMin = garbageValues.Min();
                double garbageMax = garbageValues.Max();

                Console.WriteLine($"--- Garbage Range Report ---");
                Console.WriteLine($"Safe Boundary:      {lowerOuter:N1} to {upperOuter:N1}");
                Console.WriteLine($"Actual Garbage Min: {garbageMin}"); // This will show -50
                Console.WriteLine($"Actual Garbage Max: {garbageMax}"); // This will show 498
                Console.WriteLine($"Total Garbage Count: {garbageValues.Length} rows");
            }
            else
            {
                Console.WriteLine("No garbage detected outside the Outer Fences.");
            }
        }
    }
    public class MyTools
    {
        [KernelFunction]
        [Description("This method starts Sql Server.")]
        public bool RestartSQlServer()
        {
            using var service = new ServiceController("MSSQL$SQLEXPRESS");

            // Stop if running
            if (service.Status != ServiceControllerStatus.Stopped &&
                service.Status != ServiceControllerStatus.StopPending)
            {
                service.Stop();
                service.WaitForStatus(ServiceControllerStatus.Stopped, TimeSpan.FromMinutes(2));
            }

            // Start again
            service.Start();
            service.WaitForStatus(ServiceControllerStatus.Running, TimeSpan.FromMinutes(2));
            return true;

        }

        [KernelFunction]
        [Description("Gets the most recent ERROR entry from the Windows Event Viewer Application log.")]
        public string GetCurrentEventViewerMessage(string logName = "Application")
        {
            using var eventLog = new EventLog(logName);

            for (int i = eventLog.Entries.Count - 1; i >= 0; i--)
            {
                var entry = eventLog.Entries[i];
                if (entry.EntryType == EventLogEntryType.Error)
                {
                    return $"""
                Source   : {entry.Source}
                EventId  : {entry.InstanceId}
                Time     : {entry.TimeGenerated}
                Message  :
                {entry.Message}
                """;
                }
            }

            return "No error found in Application log.";
        }
    }
}
