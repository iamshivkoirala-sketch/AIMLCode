using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace MachinelearningClass
{
    public class OnnxConsumption
    {
        public static void Execute()
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
    }

}
