using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass.InterviewQuestions
{
    public class GPT2Input
    {
        [VectorType(1, 16)] // batch=1, seq_len=16
        public long[,] input_ids { get; set; }

    }

    public class GPT2Output
    {
        [VectorType(1, 8, 50257)]
        public float[,,] logits { get; set; }
    }

    public class BertInput
    {
        [VectorType(128)]
        public long[] input_ids { get; set; }

        [VectorType(128)]
        public long[] attention_mask { get; set; }
    }

    public class BertOutput
    {
        [VectorType(384)]
        public float[] sentence_embedding { get; set; }
    }

    public class InputText
    {
        public string Text { get; set; }
    }

    public class TextEmbedding
    {
        [VectorType(384)] // embedding dimension
        public float[] Features { get; set; }
    }

    public class QAItem
    {
        public string Question { get; set; }
        public string Answer { get; set; }
        public float[] Embedding { get; set; }
    }
    public class RAGLookup
    {
        public int Id { get; set; }
        public string Description { get; set; }
        public string QuestionstobeAsked { get; set; }
        public float[] DescriptionEmbedding { get; set; }
        public List<String> Questions { get; set; }
        public RAGLookup()
        {
            Questions = new List<String>();
        }
    }
}
