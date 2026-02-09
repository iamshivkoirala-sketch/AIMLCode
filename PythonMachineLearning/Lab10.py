import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data = [
    {"Weight": 150, "Color": "Red", "FruitType": "Apple"},
    {"Weight": 130, "Color": "Green", "FruitType": "Apple"},
    {"Weight": 110, "Color": "Yellow", "FruitType": "Banana"},
    {"Weight": 180, "Color": "Yellow", "FruitType": "Banana"},
    {"Weight": 200, "Color": "Orange", "FruitType": "Orange"},
    {"Weight": 220, "Color": "Orange", "FruitType": "Orange"},
    {"Weight": 160, "Color": "Green", "FruitType": "Mango"},
    {"Weight": 170, "Color": "Yellow", "FruitType": "Mango"},
    {"Weight": 12,  "Color": "Black",  "FruitType": "Berry"},
]

df = pd.DataFrame(data)

X = df[["Weight", "Color"]]
y = df["FruitType"]

pipeline = Pipeline([
    ("preprocess", ColumnTransformer([
        ("color_enc", OneHotEncoder(), ["Color"])
    ], remainder="passthrough")),
    ("model", LogisticRegression(max_iter=1000, multi_class="multinomial"))
])
# pipeline = encoding + logisiotic
pipeline.fit(X, y) # model

test_fruit = pd.DataFrame([{"Weight": 110, "Color": "Green"}])
result = pipeline.predict(test_fruit)

print("Predicted Type:", result[0])
