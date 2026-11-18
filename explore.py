import pandas as pd

# Load the file into a table (called a DataFrame)
df = pd.read_csv("data/telco.csv")

# 1. How big is it?
print("Rows and columns:", df.shape)

# 2. What are the columns and their types?
print("\n--- Column info ---")
print(df.info())

# 3. What do the first 5 rows look like?
print("\n--- First 5 rows ---")
print(df.head())

# 4. How many customers left (churned)?
print("\n--- Churn balance ---")
print(df["Churn"].value_counts())
print(df["Churn"].value_counts(normalize=True))

# 5. Check the TotalCharges column
print("\n--- TotalCharges check ---")
print("Data type:", df["TotalCharges"].dtype)
blank = df[df["TotalCharges"].str.strip() == ""]
print("Rows with blank TotalCharges:", len(blank))