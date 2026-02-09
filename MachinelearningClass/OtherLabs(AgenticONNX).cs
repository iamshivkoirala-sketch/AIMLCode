using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.SemanticKernel;
using System.ComponentModel;
using System.ServiceProcess;
using Microsoft.SemanticKernel.ChatCompletion;
using Microsoft.SemanticKernel.Connectors.OpenAI;
using System.Diagnostics;

namespace MachinelearningClass
{
    public static class OtherLabs
    {
        public static void Lab32ConsumingONNX()
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
                    history,
                    executionSettings,
                    kernel
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
