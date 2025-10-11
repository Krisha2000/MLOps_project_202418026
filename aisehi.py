import pandas as pd

df = pd.read_csv("data/train.csv")

# rename if it’s something else
df.rename(columns={"target": "Response"}, inplace=True)

# if missing Id, add one
if "Id" not in df.columns:
    df.insert(0, "Id", range(1, len(df) + 1))

df.to_csv("data/train.csv", index=False)
print("✅ Updated train.csv with Id and Response columns")
