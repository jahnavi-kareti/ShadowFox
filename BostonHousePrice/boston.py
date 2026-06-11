import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ──────────────────────────────────────────
# STEP 1: Load Dataset
# ──────────────────────────────────────────
df = pd.read_csv('BostonHousePrice/HousingData.csv')
print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

# ──────────────────────────────────────────
# STEP 2: Check & Handle Missing Values
# ──────────────────────────────────────────
print("\nMissing values:")
print(df.isnull().sum())

# Fill missing values with median
df.fillna(df.median(), inplace=True)
print("\nAfter cleaning — missing values:", df.isnull().sum().sum())

# ──────────────────────────────────────────
# STEP 3: Exploratory Data Analysis (EDA)
# ──────────────────────────────────────────
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig('BostonHousePrice/correlation_heatmap.png')
plt.show()
print("Saved: correlation_heatmap.png")

# House price distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['MEDV'], bins=30, kde=True, color='steelblue')
plt.title('Distribution of House Prices (MEDV)')
plt.xlabel('Price ($1000s)')
plt.savefig('BostonHousePrice/price_distribution.png')
plt.show()
print("Saved: price_distribution.png")

# ──────────────────────────────────────────
# STEP 4: Prepare Features & Target
# ──────────────────────────────────────────
X = df.drop('MEDV', axis=1)
y = df['MEDV']

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTraining samples: {len(X_train)}")
print(f"Test samples    : {len(X_test)}")

# ──────────────────────────────────────────
# STEP 5: Train Models
# ──────────────────────────────────────────

# Model 1: Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# ──────────────────────────────────────────
# STEP 6: Evaluate Models
# ──────────────────────────────────────────
def evaluate(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  R²   : {r2:.4f}")
    return rmse, mae, r2

print("\n📊 Model Evaluation Results:")
lr_rmse, lr_mae, lr_r2   = evaluate("Linear Regression", y_test, lr_pred)
rf_rmse, rf_mae, rf_r2   = evaluate("Random Forest",     y_test, rf_pred)

# ──────────────────────────────────────────
# STEP 7: Visualize Predictions
# ──────────────────────────────────────────
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, lr_pred, alpha=0.6, color='steelblue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Linear Regression\nR² = {lr_r2:.4f}')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.subplot(1, 2, 2)
plt.scatter(y_test, rf_pred, alpha=0.6, color='green')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title(f'Random Forest\nR² = {rf_r2:.4f}')
plt.xlabel('Actual Price')
plt.ylabel('Predicted Price')

plt.tight_layout()
plt.savefig('BostonHousePrice/predictions_comparison.png')
plt.show()
print("Saved: predictions_comparison.png")

# ──────────────────────────────────────────
# STEP 8: Feature Importance (Random Forest)
# ──────────────────────────────────────────
feat_imp = pd.Series(rf.feature_importances_, index=df.drop('MEDV', axis=1).columns)
feat_imp = feat_imp.sort_values(ascending=False)

plt.figure(figsize=(10, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='viridis')
plt.title('Feature Importance (Random Forest)')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig('BostonHousePrice/feature_importance.png')
plt.show()
print("Saved: feature_importance.png")

print("\n🎉 Boston House Price Prediction Complete!")
print(f"Best Model: Random Forest with R² = {rf_r2:.4f}")