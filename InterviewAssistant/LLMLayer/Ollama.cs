using InterviewAssistant.Models;
using OllamaSharp;
using System;
using System.Text;
using System.Text.Json;

namespace InterviewAssistant.LLMLayer
{
    public class Ollama
    {
        public async Task<EmployeeMetaData> LoadResume(string resume)
        {
            var client = new OllamaApiClient(
                new Uri("http://localhost:11434"),
                "llama3.2"
            );
            var prompt = @$"
                Extract candidate information from the resume below.

                Rules:
                Return ONLY valid JSON.
                Do not use markdown.
                Do not use backticks.
                If a value is not found, return an empty string.
                SkillSet must always be an array.
                Education must be a string.

                Extract:
                - Name
                - Email Address
                - Contact Number
                - Location / Address / City
                - Job Title / Professional Summary
                - Education (look under sections such as EDUCATION, ACADEMIC QUALIFICATION, ACADEMIC DETAILS, QUALIFICATION, EDUCATIONAL BACKGROUND)
                - Skills

                JSON Format:
                {{
                ""Name"": """",
                ""EmailAddress"": """",
                ""ContactNumber"": """",
                ""Location"": """",
                ""JobDescription"": """",
                ""SkillSet"": [],
                ""Education"": """"
                }}

                Resume:
                {resume}
            ";

            var responseText = new StringBuilder();
            await foreach (var response in client.GenerateAsync(prompt))
            {
                responseText.Append(response.Response);
            }
            try
            {
                var json = responseText.ToString()
                    .Replace("json", "")
                    .Replace("```", "")
                    .Trim();

                var employee = JsonSerializer.Deserialize<EmployeeMetaData>(
                    json,
                    new JsonSerializerOptions
                    {
                        PropertyNameCaseInsensitive = true
                    });
                return employee ?? new EmployeeMetaData();
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error parsing response: {ex.Message}");
                return new EmployeeMetaData();
            }
        }
    }
}

