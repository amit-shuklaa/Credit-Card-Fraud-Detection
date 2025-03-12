import pandas as pd

df = pd.read_csv("creditcard.csv")

# Filter fraud cases (Class = 1)
fraud_cases = df[df["Class"] == 1]

# Display the first few fraud transactions
print(fraud_cases.head())
print(df.iloc[541])