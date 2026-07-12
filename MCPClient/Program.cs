using Microsoft.Extensions.AI;
using ModelContextProtocol.Client;
using OllamaSharp;
using ModelContextProtocol;
using OpenAI.Chat;
using MachinelearningClass;

namespace MCPClient
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            var transport = new StdioClientTransport(new StdioClientTransportOptions
            {
                Name = "Greetingtools",
                Command = "dotnet",
                Arguments =
                [
                    @"C:\Users\shivB\source\repos\MachinelearningClass\MCPServerDotNet\bin\Debug\net8.0\MCPServerDotNet.dll"
                ]
            });
            // here to 
            await using var mcpClient = await McpClient.CreateAsync(transport);
            var mcpTools = await mcpClient.ListToolsAsync();

            Console.WriteLine("Available MCP Tools:");
            foreach (var tool in mcpTools)
            {
                Console.WriteLine($"- {tool.Name}");
            }
            // here we are talking with the MCP server and knowing the tools.
            //IChatClient chatClient = new OllamaApiClient(
            //    new Uri("http://localhost:11434"),
            //    "llama3.2"    
            //);

            //IChatClient client = new OllamaChatClient(new Uri("http://localhost:11434"), "llama3.2")
            //                 .AsBuilder()
            //                 .UseFunctionInvocation()
            //                 //.UseFunctionInvocation(configure:options => {

            //                 //    options.MaximumIterationsPerRequest = 1;
            //                 //})
            //                 .Build();
            var key = Environment.GetEnvironmentVariable("aikey");

            IChatClient client = new ChatClient("gpt-4o-mini", key)
            .AsIChatClient() // Bridges OpenAI.Chat to IChatClient
            .AsBuilder()
            //.UseFunctionInvocation()
            //.UseFunctionInvocation(configure: options =>
            //{

            //    options.MaximumIterationsPerRequest = 1;
            //})
            .Build();
            while (true)
            {
                Console.WriteLine("Enter name");
                string name = Console.ReadLine();
                //var prompt = $"Time: {DateTime.Now}. Name: {name}. Depending on " +  DateTime.Now
                //+ " decide its morning , evening or afternoon." +
                //                "and call the functions/tools goodmorning / goodevening / goodafternoon " +
                //                " pass the name to the tool methods";
                //"if";
                //var prompt = $"The person's name is {name}. The current time is {DateTime.Now}. " +
                // "Select and call the single most appropriate greeting tool for this time of day." +
                // "As per its evening , morning , night call the method names"  ;
                var prompt = $"The person's name is {name}.  " +
                 "Select and call the single most appropriate greeting tool for this time of day." +
                 "As per the current date and time if its evening , morning , night call the methods" +
                 "of the tool accordingly" +
                 "pass the person name to the method and invoke it";
              
                var response = await client.GetResponseAsync(
                 prompt,
                   new ChatOptions
                   {
                       Tools = new List<AITool>(mcpTools),
                       ToolMode = ChatToolMode.RequireAny
                   }
                );

                Console.WriteLine("\nAI Response:");
                Console.WriteLine(response.Text);
            }
        }
    }
}