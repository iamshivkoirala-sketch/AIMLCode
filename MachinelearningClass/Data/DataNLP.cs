using MachinelearningClass.InterviewQuestions;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass.DataNLP
{
    public static class DataforNlp
    {
        public static List<RAGLookup> getRAGData()
        {
            return new List<RAGLookup>
            {
            new RAGLookup
            {
                Description = "0-3 yrs year experience .NET developer",
                QuestionstobeAsked = "Basics of C#, OOP, SQL Server."
            },
            new RAGLookup
            {
                Description = "Senior .NET developer with 5+ years",
                QuestionstobeAsked = "Advanced ASP.NET Core, microservices, cloud, design patterns."
            },
            new RAGLookup
            {
                Description = "Architect 10+ years",
                QuestionstobeAsked = "Distributed systems, DDD, Azure architecture, performance tuning."
            }
            };
        }
     
    }
}
