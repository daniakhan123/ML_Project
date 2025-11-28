
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore")

DATA_PATH = "vehicle_price_prediction.csv"
CURRENT_YEAR = 2025
CHUNK_SIZE = 200000


print("Loading data...")
chunks = []

sample_df = pd.read_csv(DATA_PATH, nrows=5)
print("Dataset columns:", sample_df.columns.tolist())
print("Dataset sample shape:", sample_df.shape)

for chunk in pd.read_csv(DATA_PATH, chunksize=CHUNK_SIZE):
    if len(chunk) > 1000:
        chunks.append(chunk.sample(frac=0.1, random_state=42))
    else:
        chunks.append(chunk)

df = pd.concat(chunks, ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle
print(f"Final dataset size: {df.shape}")


print("\n=== DATA EXPLORATION ===")
print("First 5 rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nBasic stats:")
print(df.describe(include='all'))


print("\n=== HANDLING MISSING VALUES ===")
print("Missing values before:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")


if numeric_cols:
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
if categorical_cols:
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])

df = df.drop_duplicates().reset_index(drop=True)
print("Missing values after (should be zeros):")
print(df.isnull().sum())


print("\n=== FEATURE ENGINEERING ===")
if 'year' in df.columns:
    df['Car_Age'] = CURRENT_YEAR - df['year']
    df.drop('year', axis=1, inplace=True)
    print("Created Car_Age")
else:
    print("WARNING: 'year' column not found!")

if 'make' in df.columns:
    luxury_brands = ['Mercedes', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Jaguar', 'Land Rover']
    df['Is_Luxury'] = df['make'].apply(lambda x: 1 if x in luxury_brands else 0)
    df.drop('make', axis=1, inplace=True)
    print("Created Is_Luxury")
else:
    print("WARNING: 'make' column not found!")


if 'price' not in df.columns:
    raise ValueError("Dataset must contain 'price' column.")
original_size = len(df)
df = df[(df['price'] > df['price'].quantile(0.01)) & (df['price'] < df['price'].quantile(0.99))]
print(f"Removed {original_size - len(df)} price outliers")

df['original_price'] = df['price']
df['price'] = np.log1p(df['price'])
print(f"Price transformed: log1p applied (target is now 'price')")


categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical cols to encode: {categorical_cols}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le
    print(f"Encoded {col} ({len(le.classes_)} classes)")


numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
# exclude target and binary markers from scaling
numeric_to_scale = [c for c in numeric_cols if c not in ['price', 'Is_Luxury', 'original_price']]
print(f"Numeric columns to scale: {numeric_to_scale}")

scaler = StandardScaler()
if numeric_to_scale:
    df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
    print("Applied StandardScaler to numeric features")
else:
    print("No numeric columns to scale (unexpected)")

X = df.drop(['price', 'original_price'], axis=1)
y = df['price']

print(f"\nFinal feature matrix shape: {X.shape}")
print("Feature names:", X.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n=== TRAINING MULTIPLE MODELS ===")

models = {
    "Linear Regression": LinearRegression(),
    "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
    "Decision Tree": DecisionTreeRegressor(max_depth=10, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=15, random_state=42, n_jobs=-1),
    "Gradient Boosting (Ensemble)": GradientBoostingRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
}

models["Simple ANN"] = MLPRegressor(
    hidden_layer_sizes=(10,),
    activation='relu',
    solver='adam',
    max_iter=500,
    random_state=42
)


results = {}
trained_models = {}

for name, model in models.items():
    print(f"\nTraining: {name}")
    model.fit(X_train, y_train)
    trained_models[name] = model

    y_pred_log = model.predict(X_test)


    y_test_actual = np.expm1(y_test)
    y_pred_actual = np.expm1(y_pred_log)

    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
    mae = mean_absolute_error(y_test_actual, y_pred_actual)
    r2 = r2_score(y_test_actual, y_pred_actual)

    results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"{name} → RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")


results_df = pd.DataFrame(results).T
print("\n=== Model comparison (sorted by R² desc) ===")
print(results_df.sort_values('R2', ascending=False))

best_name = results_df['R2'].idxmax()
best_model = trained_models[best_name]
print(f"\nSelected best model: {best_name} (highest R²)")


y_pred_log_best = best_model.predict(X_test)
y_test_actual = np.expm1(y_test)
y_pred_actual_best = np.expm1(y_pred_log_best)

best_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual_best))
best_mae = mean_absolute_error(y_test_actual, y_pred_actual_best)
best_r2 = r2_score(y_test_actual, y_pred_actual_best)
print(f"{best_name} final metrics → RMSE: {best_rmse:.2f}, MAE: {best_mae:.2f}, R²: {best_r2:.4f}")

def improved_feature_analysis(model, X_train, X_test, feature_names):
    """Analyze average effect of +/-20% on key features for several test samples."""
    print("\n=== FEATURE RELATIONSHIP ANALYSIS ===")
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)

    key_features = ['mileage', 'engine_hp', 'engineSize', 'Car_Age']
    available = [f for f in key_features if f in feature_names]

    for feature in available:
        print(f"\n--- {feature} ---")
        effects = []
        for idx in sample_indices:
            test_sample = X_test.iloc[idx:idx+1].copy()
            baseline_pred = model.predict(test_sample)[0]
            baseline_price = np.expm1(baseline_pred)

            original_val = test_sample[feature].values[0]

      
            test_sample[feature] = original_val * 1.2
            pred_inc = model.predict(test_sample)[0]
            price_inc = np.expm1(pred_inc)

            test_sample[feature] = original_val * 0.8
            pred_dec = model.predict(test_sample)[0]
            price_dec = np.expm1(pred_dec)

            eff_inc = (price_inc - baseline_price) / baseline_price
            eff_dec = (price_dec - baseline_price) / baseline_price
            effects.append((eff_inc, eff_dec))

        avg_inc = np.mean([e[0] for e in effects])
        avg_dec = np.mean([e[1] for e in effects])
        print(f"Avg effect of +20%: {avg_inc:+.2%}, Avg effect of -20%: {avg_dec:+.2%}")

improved_feature_analysis(best_model, X_train, X_test, X.columns.tolist())


print("\n=== FEATURE IMPORTANCE ===")
if hasattr(best_model, "feature_importances_"):
    importances = best_model.feature_importances_
    features = X.columns
    fi_df = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance', ascending=False)
    print(fi_df.head(15))
    plt.figure(figsize=(10,6))
    sns.barplot(data=fi_df.head(15), x='importance', y='feature')
    plt.title(f"Top Features - {best_name}")
    plt.tight_layout()
    plt.show()
else:
    
    if hasattr(best_model, "coef_"):
        coef = best_model.coef_
        features = X.columns
        coef_df = pd.DataFrame({'feature': features, 'coef': coef}).sort_values('coef', key=lambda s: np.abs(s), ascending=False)
        print(coef_df.head(15))
        plt.figure(figsize=(10,6))
        sns.barplot(data=coef_df.head(15), x='coef', y='feature')
        plt.title(f"Top Coefficients - {best_name}")
        plt.tight_layout()
        plt.show()
    else:
        print(f"No feature importance / coef available for model type: {type(best_model)}")

plt.figure(figsize=(8,6))
plt.scatter(y_test_actual, y_pred_actual_best, alpha=0.4)
plt.plot([y_test_actual.min(), y_test_actual.max()],
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted - {best_name}")
plt.tight_layout()
plt.show()


print("\n=== SAVING ARTIFACTS ===")
pickle.dump(best_model, open("best_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

print("Saved: best_model.pkl, scaler.pkl, label_encoders.pkl, feature_names.pkl")
print("Pipeline complete.")
