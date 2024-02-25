import os
import pandas as pd

# made a "FILES.CSV" using something like dir /a /s /b > FILES.csv in the command prompt of the folder in question
df = pd.read_csv("FILES.csv", header=None)
df[0] = df[0].str.replace(
    os.getcwd() + "\\train\\",
    "",
)
df = df[:-1]
df[1] = 0
df.loc[df[0].str.contains("dog"), 1] = 1
print(df)
df.to_csv("train.csv", index=False, header=False)

# similarly for the testing set but don't add a label column
