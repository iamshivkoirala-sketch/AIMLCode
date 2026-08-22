using ModelContextProtocol.Server;
using System.ComponentModel;

namespace MCPServerHttp
{
    public class Program
    {
        public static void Main(string[] args)
        {
            var builder = WebApplication.CreateBuilder(args);

            // Existing MVC services
            builder.Services.AddControllersWithViews();

            // Add MCP services middle ware
            builder.Services
                .AddMcpServer()
                .WithHttpTransport(options =>
                {
                    options.Stateless = true;
                })
                .WithToolsFromAssembly();

            var app = builder.Build();

            if (!app.Environment.IsDevelopment())
            {
                app.UseExceptionHandler("/Home/Error");
                app.UseHsts();
            }

            app.UseHttpsRedirection();
            app.UseStaticFiles();

            app.UseRouting();
            app.UseAuthorization();

            // Existing MVC endpoint
            app.MapControllerRoute(
                name: "default",
                pattern: "{controller=Home}/{action=Index}/{id?}");

            // MCP endpoint
            app.MapMcp("/mcp");

            app.Run();
        }
    }
    [McpServerToolType]
    public class GreetingTool
    {
        [McpServerTool(Name = "GoodMorning")]
        [Description("Call this if some one sayes good morning , gm and any other language ")]
        public string GoodMorning([Description("Name of the user")] string name)
        {
            return $"Good Morning, {name}.";
        }

        [McpServerTool(Name = "GoodEvening")]
        [Description("Call this if some one sayes good evening , GE")]
        public string GoodEvening([Description("Name of the user")] string name)
        {
            return $"Good Evening, {name}. ";
        }

        [McpServerTool(Name = "GoodAfternoon")]
        [Description("Call this if some one sayes good afternoon , happy noon")]
        public string GoodAfternoon([Description("Name of the user")] string name)
        {

            return $"Good Afternoon, {name}. ";
        }
    }
}