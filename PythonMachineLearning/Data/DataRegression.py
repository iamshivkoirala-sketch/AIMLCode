from MachineLearning.Model.InsuranceModel import InsuranceData


class Data :
    @staticmethod
    def GetLinearInsuranceData():
        return [
             InsuranceData(10, 3000),
            InsuranceData(20, 4010),
            InsuranceData(30, 5020),
            InsuranceData(40, 6000),
            InsuranceData(50, 7000),
            InsuranceData(60, 8900),
            InsuranceData(70, 9000),
            InsuranceData(80, 10000),
            InsuranceData(90, 11000),
            InsuranceData(100, 12000),
            
        ]