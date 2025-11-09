import pandas as pd

# Load processed features
X = pd.read_csv('data/processed/features_X_20251109.csv', index_col=0)
y = pd.read_csv('data/processed/target_y_20251109.csv', index_col=0)

# Check for missing values
print("Missing values in X:", X.isnull().sum().sum())
print("Missing values in y:", y.isnull().sum().sum())

# Check feature statistics
print("\nFeature statistics:")
print(X.describe())

# Check target statistics
print("\nTarget statistics:")
print(y.describe())

# Correlation with target
print("\nFeature correlations with target:")
correlations = X.corrwith(y.squeeze()).sort_values(ascending=False)
print(correlations.head(10))  # Top 10 most correlated features