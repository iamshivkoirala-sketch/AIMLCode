

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using LangChain.Providers;
using LangChain.Splitters.Text;
using Microsoft.SemanticKernel.Text;
using Neo4j.Driver;
using OpenAI;
using OpenAI.Chat;
namespace MachinelearningClass
{
    public static class ChunkingLessons
    {
        public static void FixedAndRecursiveChunking()
        {
            string data =
                 "C# is a modern object-oriented programming language developed by Microsoft.\n\n" +
                 "It supports features such as classes, interfaces, inheritance, and generics.\n\n" +
                 "C# is widely used for building web applications, desktop applications," +
                 "cloud services, and mobile apps using the .NET platform." +
                 "What is WCF ?" +
                 "Azure Service Bus is a messaging service." +
                 "It is used for asynchronous communication.";
            //string data = "Q: What is Azure?\n" +
            //                        "A: Azure is a cloud platform.\n\n" +
            //                        "Q: What is .NET?\n" +
            //                        "A: .NET is a development platform.";
            //var splitter = new CharacterTextSplitter(
            //                separator: " ",
            //                chunkSize: 70,
            //                chunkOverlap: 20);

            var splitter = new RecursiveCharacterTextSplitter(
                    chunkSize: 70,
                    chunkOverlap: 20);
            var chunks = splitter.SplitText(data);

            foreach (var chunk in chunks)
            {
                Console.WriteLine("**********************");
                Console.WriteLine(chunk);
            }

        }
        public static async Task SemanticChunkingDemo()
    {
            string[] lines =
            {
            "C# supports classes and interfaces.",
            "C# supports inheritance and polymorphism.",
            "Azure Service Bus is a messaging service.",
            "Azure Service Bus supports queues and topics.",
            "SQL Server stores relational data."
             };

            var client = new OpenAIClient(
                Environment.GetEnvironmentVariable("aikey"));

             var embeddings = new List<float[]>(); // list of float

            foreach (var line in lines)
            {
                var response =
                    await client
                        .GetEmbeddingClient("text-embedding-3-small")
                        .GenerateEmbeddingAsync(line); // creating the vector

                embeddings.Add(
                    response.Value.ToFloats().ToArray()); // addin to the embedding
            }

            var chunks = new List<List<string>>();

            var currentChunk = new List<string>
            {
                lines[0]
            };

            for (int i = 1; i < lines.Length; i++)
            {
                float[] v1 = embeddings[i - 1];
                float[] v2 = embeddings[i];

           
                if (Common.CalculateCosineSimilarity(v1,v2) > 0.50)
                {
                    currentChunk.Add(lines[i]);
                }
                else
                {
                    chunks.Add(currentChunk);

                    currentChunk = new List<string>
                {
                    lines[i]
                };
                }
            }

            chunks.Add(currentChunk);

       
        Console.WriteLine("Semantic Chunks");

        foreach (var chunk in chunks)
        {
            Console.WriteLine("Same Semantic-------------");

            foreach (var line in chunk)
            {
                Console.WriteLine(line);
            }
        }
    }
        public static async Task AgenticSemanticChunkingDemo()
            {
                string[] lines =
                {
                "C# supports classes and interfaces.",
                "C# supports inheritance and polymorphism.",
                "Azure Service Bus is a messaging service.",
                "Azure Service Bus supports queues and topics.",
                "SQL Server stores relational data."
            };

                var client = new OpenAIClient(
                    Environment.GetEnvironmentVariable("aikey"));

                var chatClient = client.GetChatClient("gpt-4o-mini");

                string inputText = string.Join("\n", lines);

                string prompt = $@"
                        You are an AI chunking agent.

                        Group the following lines into semantic chunks.

                        Rules:
                        - Keep similar meaning together.
                        - Do not change the original lines.
                        - Do not invent new lines.
                        - Return valid JSON only.

                        JSON format:
                        {{
                          ""chunks"": [
                            {{
                              ""title"": ""chunk title"",
                              ""reason"": ""why these lines are grouped"",
                              ""lines"": [""line 1"", ""line 2""]
                            }}
                          ]
                        }}

                        Lines:
                        {inputText}
                        ";

                ChatCompletion response =
                    await chatClient.CompleteChatAsync(prompt);

                string json = response.Content[0].Text;

                Console.WriteLine(json);
            }
        public static async Task EntityBasedChunkingUsingLLM()
        {
            string[] lines =
            {
        "C# supports classes and interfaces.",
        "C# supports inheritance and polymorphism.",
        "Azure Service Bus is a messaging service.",
        "Azure Service Bus supports queues and topics.",
        "SQL Server stores relational data."
    };

            var client = new OpenAIClient(
                Environment.GetEnvironmentVariable("aikey"));

            var chatClient = client.GetChatClient("gpt-4o-mini");

            string text = string.Join("\n", lines);

            string prompt = $@"
        You are an entity-based chunking assistant.

        Task:
        Extract important entities from the text and group related lines under each entity.

        Rules:
        - Do not invent new content.
        - Keep original lines unchanged.
        - One line can belong to only one main entity.
        - Return valid JSON only.

        JSON format:
        {{
          ""entityChunks"": [
            {{
              ""entityName"": ""C#"",
              ""entityType"": ""Technology"",
              ""lines"": [
                ""original line 1"",
                ""original line 2""
              ]
            }}
          ]
        }}

        Text:
        {text}
        ";

            ChatCompletion response =
                await chatClient.CompleteChatAsync(prompt);

            string json = response.Content[0].Text;

            Console.WriteLine(json);
        }
        public static async Task InsertDataNeo4j()
        {
            var driver = GraphDatabase.Driver(
                "bolt://localhost:7687",
                AuthTokens.Basic("neo4j", "pass@123"));

            await using var session = driver.AsyncSession(o => o.WithDatabase("neo4j"));

            await session.ExecuteWriteAsync(async tx =>
            {
                await tx.RunAsync(@"
            CREATE (p:Person {name:'Shiv'})
            CREATE (s:City {name:'Mumbai'})
            CREATE (p)-[:Stays]->(s)
            ");
            });

            
        }
        public static async Task DisplayDataNeo4j()
        {
            await using var driver = GraphDatabase.Driver(
                "bolt://localhost:7687",
                AuthTokens.Basic("neo4j", "password@123"));

            await driver.VerifyConnectivityAsync();

            await using var session =
                driver.AsyncSession(o => o.WithDatabase("neo4j"));

            var result = await session.RunAsync(@"
        MATCH (p:Person)-[:KNOWS]->(s:Skill)
        RETURN p.name AS Person,
               s.name AS Skill
    ");

            await foreach (var record in result)
            {
                Console.WriteLine(
                    $"{record["Person"]} knows {record["Skill"]}");
            }
        }
    }

  

   

}
