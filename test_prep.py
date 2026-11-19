from src.preprocess import CATEGORICAL_FEATURES, FEATURES, TARGET, build_preprocessor, load_data

df = load_data("data/telco.csv")
X = df[FEATURES]

print("Shape:", df.shape)
print("Churn rate:", round(df[TARGET].mean(), 3))
print("Missing values:", int(df.isna().sum().sum()))

X_ready = build_preprocessor().fit_transform(X)
print("After preprocessing:", X_ready.shape)

print("\nCategory values (we need these for the API later):")
for c in CATEGORICAL_FEATURES:
    print(c, "->", sorted(df[c].unique()))