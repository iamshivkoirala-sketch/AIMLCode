using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public static  class Data
    {
         public class CustomerData
        {
            public float Age { get; set; }
            public float Spending { get; set; }
        }

        public class FruitData
        {
            public float Weight { get; set; }
            public string Color { get; set; }
            public bool IsApple { get; set; }
            public string FruitType { get; set; }
        }

        public class InsuranceData
        {
            public float Age { get; set; }
            public float Premium { get; set; }
        }

        public static List<CustomerData> GetCustomerData()
        {
            var customers = new[]
            {
                new CustomerData { Age = 22, Spending = 20000 },
                new CustomerData { Age = 25, Spending = 23000 },
                new CustomerData { Age = 45, Spending = 40000 },
                new CustomerData { Age = 50, Spending = 42000 },
                new CustomerData { Age = 65, Spending = 15000 },
                new CustomerData { Age = 70, Spending = 12000 },
            };
            return customers.ToList<CustomerData>();
        }

        public static List<FruitData> GetFruitData()
        {
            var samples = new[]
            {
            new FruitData { Weight = 150, Color = "Red",   IsApple = true,  FruitType = "Apple" },
            new FruitData { Weight = 130, Color = "Green", IsApple = true,  FruitType = "Apple" },
            new FruitData { Weight = 110, Color = "Yellow",IsApple = false, FruitType = "Banana" },
            new FruitData { Weight = 180, Color = "Yellow",IsApple = false, FruitType = "Banana" },
            new FruitData { Weight = 200, Color = "Orange",IsApple = false, FruitType = "Orange" },
            new FruitData { Weight = 220, Color = "Orange",IsApple = false, FruitType = "Orange" },
            new FruitData { Weight = 160, Color = "Green", IsApple = false, FruitType = "Mango" },
            new FruitData { Weight = 170, Color = "Yellow",IsApple = false, FruitType = "Mango" },
            new FruitData { Weight = 12, Color = "Black",IsApple = false, FruitType = "Berry" },

            };

            return samples.ToList();
        }
        public static List<InsuranceData> GetLinearInsuranceData()
        {
            return new List<InsuranceData>
            {
                new InsuranceData { Age = 10, Premium = 2000 },
                new InsuranceData { Age = 20, Premium = 4000 },
                new InsuranceData { Age = 30, Premium = 6000 },
                new InsuranceData { Age = 40, Premium = 8000 },
                new InsuranceData { Age = 50, Premium = 10000 },
                new InsuranceData { Age = 60, Premium = 12000 },
                new InsuranceData { Age = 70, Premium = 14000 },
                new InsuranceData { Age = 80, Premium = 16000 },
                new InsuranceData { Age = 90, Premium = 18000 },
                new InsuranceData { Age = 100, Premium = 20000 }
            };
        }
        public  static List<InsuranceData> GetTestData()
        {
            return new List<InsuranceData>
                {
                    new InsuranceData { Age = 80, Premium = 16000 },
                    new InsuranceData { Age = 90, Premium = 18000 },
                    new InsuranceData { Age = 100, Premium = 20000 }
                };
        }
    }
}
