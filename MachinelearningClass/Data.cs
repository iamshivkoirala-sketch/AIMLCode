using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace MachinelearningClass
{
    public static  class Data
    {
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
