using Microsoft.Extensions.DependencyInjection; 
using Microsoft.Extensions.Hosting; // Lets you create and run a host application
using Microsoft.Extensions.Logging; // Lets you log messages
using ModelContextProtocol.Server; // Provides MCPServer features
using System.ComponentModel; // Lets you add descriptions to method parameters
using Microsoft.AspNetCore.Builder;
namespace MCPServerDotNet
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            var builder = Host.CreateApplicationBuilder(args); // Creates a builder to sets up the app environment
            builder.Logging.ClearProviders(); // Removes default logging providers

            builder.Services
                .AddMcpServer() // Adds MCPServer services
                .WithStdioServerTransport() // Uses standard input/output for communication
                .WithToolsFromAssembly(); // Finds and registers all tool classes in the assembly

            var app = builder.Build(); // Builds the application
            Console.WriteLine("MCP server running");
            //Console.Read();
            await app.RunAsync(); // Runs the application asynchronously
            Console.WriteLine("MCP server running");
            Console.Read();
        }
        
        
    }
    [McpServerToolType] 
    public class MyTools
    {
        [McpServerTool(Name = "GoodMorning")]
        [Description("Call this if the current time is between 5 PM and 4:59 AM. ")]
        public string GoodMorning([Description("Name of the user")] string name)
        {
            return $"Good Morning, {name}. give a proverb";
        }

        [McpServerTool(Name = "GoodEvening")]
        [Description("Call this if the current time is between 5 AM and 11:59 AM. Send output with out any modification")]
        public string GoodEvening([Description("Name of the user")] string name)
        {
            return $"Good Evening, {name}. give a proverb";
        }

        [McpServerTool(Name = "GoodAfternoon")]
        [Description("Call this if the current time is between 12 PM and 4:59 PM ")]
        public string GoodAfternoon([Description("Name of the user")] string name)
        {
            
            return "Good Afternoon ...." + name + " give a proverb";
        }
    }
}
