import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb


print("Loading data...")
chunk_size = 200000
chunks = []

sample_df = pd.read_csv("vehicle_price_prediction.csv", nrows=5)
print("Dataset columns:", sample_df.columns.tolist())
print("Dataset shape sample:", sample_df.shape)

for chunk in pd.read_csv("vehicle_price_prediction.csv", chunksize=chunk_size):
    if len(chunk) > 1000:  
        chunks.append(chunk.sample(frac=0.1, random_state=42))
    else:
        chunks.append(chunk)

df = pd.concat(chunks)
df = df.sample(frac=1, random_state=42) 

print(f"Final dataset size: {df.shape}")

print("\n=== DATA EXPLORATION ===")
print("First 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

print("\nAll column names:")
print(df.columns.tolist())

print("\n=== HANDLING MISSING VALUES ===")
print("Missing values before:")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include='object').columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
df = df.drop_duplicates()

print("Missing values after:")
print(df.isnull().sum())


print("\n=== FEATURE ENGINEERING ===")

if 'year' in df.columns:
    CURRENT_YEAR = 2025
    df['Car_Age'] = CURRENT_YEAR - df['year']
    df.drop('year', axis=1, inplace=True)
    print("Created Car_Age feature")
else:
    print("WARNING: 'year' column not found!")

if 'make' in df.columns:
    luxury_brands = ['Mercedes', 'BMW', 'Audi', 'Lexus', 'Porsche', 'Jaguar', 'Land Rover']
    df['Is_Luxury'] = df['make'].apply(lambda x: 1 if x in luxury_brands else 0)
    df.drop('make', axis=1, inplace=True)
    print("Created Is_Luxury feature")
else:
    print("WARNING: 'make' column not found!")

original_size = len(df)
df = df[(df['price'] > df['price'].quantile(0.01)) & 
        (df['price'] < df['price'].quantile(0.99))]
print(f"Removed {original_size - len(df)} price outliers")

print(f"Price statistics before log transform: min={df['price'].min():.2f}, max={df['price'].max():.2f}, mean={df['price'].mean():.2f}")
df['original_price'] = df['price'] 
df['price'] = np.log1p(df['price'])
print(f"Price statistics after log transform: min={df['price'].min():.2f}, max={df['price'].max():.2f}, mean={df['price'].mean():.2f}")

categorical_cols = df.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns to encode: {categorical_cols}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le
    print(f"Encoded {col} with {len(le.classes_)} categories")

numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
print(f"All numeric columns: {numeric_cols}")

numeric_to_scale = [col for col in numeric_cols 
                   if col not in ['price', 'Is_Luxury', 'original_price'] and col in df.columns]

print(f"Columns to scale: {numeric_to_scale}")

scaler = StandardScaler()
if numeric_to_scale:
    df[numeric_to_scale] = scaler.fit_transform(df[numeric_to_scale])
    print("Applied scaling to numeric features")
else:
    print("WARNING: No numeric columns to scale!")

def comprehensive_validation(df):
    """Comprehensive data validation"""
    
    print("\n=== COMPREHENSIVE DATA VALIDATION ===")

    numeric_cols_for_corr = [col for col in df.select_dtypes(include=np.number).columns 
                           if col != 'original_price']
    
    corr_matrix = df[numeric_cols_for_corr].corr()
    print("Correlation with price (all numeric features):")
    price_correlations = corr_matrix['price'].sort_values(ascending=False)
    print(price_correlations)

    key_features = ['mileage', 'engine_hp', 'engineSize', 'mpg', 'tax', 'Car_Age']
    available_features = [f for f in key_features if f in df.columns]
    
    print(f"\nAvailable key features: {available_features}")
    
    for feature in available_features:
        if feature in df.columns:
            correlation = df['price'].corr(df[feature])
            print(f"Price vs {feature} correlation: {correlation:.3f}")
    
   
    n_features = len(available_features)
    if n_features > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for i, feature in enumerate(available_features[:6]):
            if i < len(axes):
                axes[i].scatter(df[feature], df['original_price'], alpha=0.5)
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Original Price')
                axes[i].set_title(f'Price vs {feature}\n(corr: {df["original_price"].corr(df[feature]):.3f})')
        
        for i in range(n_features, 6):
            if i < len(axes):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    return price_correlations

correlations = comprehensive_validation(df)


print("\n=== DATA QUALITY CHECKS ===")


constant_cols = []
for col in df.columns:
    if df[col].nunique() <= 1:
        constant_cols.append(col)

if constant_cols:
    print(f"WARNING: Constant columns found: {constant_cols}")
    df = df.drop(columns=constant_cols)

numeric_df = df.select_dtypes(include=np.number)
if len(numeric_df.columns) > 1:
    high_corr_pairs = []
    corr_matrix = numeric_df.corr().abs()
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if corr_matrix.iloc[i, j] > 0.95:
                high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        print("Highly correlated feature pairs (>0.95):")
        for pair in high_corr_pairs:
            print(f"  {pair[0]} vs {pair[1]}: {pair[2]:.3f}")


X = df.drop(['price', 'original_price'], axis=1)
y = df['price']

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print("\n=== TRAINING MODEL ===")
lgb_model = lgb.LGBMRegressor(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.05,
    num_leaves=31,
    n_jobs=-1,
    random_state=42
)

lgb_model.fit(X_train, y_train)


y_pred_log = lgb_model.predict(X_test)


y_test_actual = np.expm1(y_test)
y_pred_actual = np.expm1(y_pred_log)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print(f"\n=== MODEL PERFORMANCE ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.2f}")


def improved_feature_analysis(model, X_train, X_test, feature_names):
    """Comprehensive feature relationship analysis"""
    
    print("\n=== FEATURE RELATIONSHIP ANALYSIS ===")
    
   
    sample_indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
    
    for feature in ['mileage', 'engine_hp', 'engineSize', 'Car_Age']:
        if feature in feature_names:
            print(f"\n--- {feature.upper()} Analysis ---")
            
            effects = []
            for idx in sample_indices:
                test_sample = X_test.iloc[idx:idx+1].copy()
                baseline_pred = model.predict(test_sample)[0]
                baseline_price = np.expm1(baseline_pred)
                
                # Store original value
                original_val = test_sample[feature].values[0]
                
                # Test increase
                test_sample[feature] = original_val * 1.2  # 20% increase
                pred_increased = model.predict(test_sample)[0]
                price_increased = np.expm1(pred_increased)
                
                # Test decrease
                test_sample[feature] = original_val * 0.8  # 20% decrease
                pred_decreased = model.predict(test_sample)[0]
                price_decreased = np.expm1(pred_decreased)
                
                effect_increase = (price_increased - baseline_price) / baseline_price
                effect_decrease = (price_decreased - baseline_price) / baseline_price
                
                effects.append((effect_increase, effect_decrease))
            
            avg_effect_inc = np.mean([e[0] for e in effects])
            avg_effect_dec = np.mean([e[1] for e in effects])
            
            print(f"Average effect of +20% {feature}: {avg_effect_inc:+.2%}")
            print(f"Average effect of -20% {feature}: {avg_effect_dec:+.2%}")
            
            if 'mileage' in feature.lower():
                expected_sign = '-'  # Higher mileage should decrease price
            elif 'hp' in feature.lower() or 'engine' in feature.lower():
                expected_sign = '+'  # Higher HP should increase price
            elif 'age' in feature.lower():
                expected_sign = '-'  # Older cars should be cheaper
            else:
                expected_sign = '?'
            
            actual_sign_inc = '+' if avg_effect_inc > 0 else '-'
            matches_expectation_inc = (actual_sign_inc == expected_sign)
            
            print(f"Expected sign: {expected_sign}, Actual sign: {actual_sign_inc}, Matches: {matches_expectation_inc}")

improved_feature_analysis(lgb_model, X_train, X_test, X.columns.tolist())


importances = lgb_model.feature_importances_
features = X.columns

feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n=== FEATURE IMPORTANCE ===")
print(feature_importance_df.head(10))

plt.figure(figsize=(10,6))
sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
plt.title("Top 15 Feature Importances")
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(y_test_actual, y_pred_actual, alpha=0.5)
plt.plot([y_test_actual.min(), y_test_actual.max()], 
         [y_test_actual.min(), y_test_actual.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


print("\n=== SAVING ARTIFACTS ===")
pickle.dump(lgb_model, open("lgb_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(le_dict, f)
with open("feature_names.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)
