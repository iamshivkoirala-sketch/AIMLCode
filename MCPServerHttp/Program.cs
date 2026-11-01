using InterviewTools;
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
                //.WithToolsFromAssembly()
                .WithTools<EmailTool>();

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
  
   
}