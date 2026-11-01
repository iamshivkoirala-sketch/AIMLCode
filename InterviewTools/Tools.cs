using ModelContextProtocol.Server;
using System.ComponentModel;

namespace InterviewTools
{
    [McpServerToolType]
    public class EmailTool
    {
        
        [McpServerTool(Name = "SendEmail")]
        [Description("Use this  tool to send email to the developer who is taking the test ")]
        public static string SendEmail(
        [Description("email address")] string email)
        {
            // roles
            Console.WriteLine("You have started the interview at " + DateTime.Now.ToString() +
                                " and we are sending email at " + email);
            return "You have started the interview at ... " + DateTime.Now.ToString() +
                                " and we are sending email at " + email;

        }
    }
}
