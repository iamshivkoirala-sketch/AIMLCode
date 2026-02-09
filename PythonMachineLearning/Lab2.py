from MachineLearning.Data.DataRegression import Data
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame([(d.Age , d.Premium) for d in
        Data.GetLinearInsuranceData()], 
        columns=["Age", "Premium"]
        )
y = df["Premium"] # Label
x = df[["Age"]] # feature
modelx = LinearRegression()
modelx.fit(x,y)

df["predicted"] = modelx.predict(x)
print(df["predicted"])
