import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Load data
print("Loading data...")
df = pd.read_excel('contagious_data.xlsx', header=0)

# Ensure all column names are strings
df.columns = df.columns.astype(str)

# Check for an empty DataFrame
if df.empty:
    raise ValueError("The DataFrame is empty. Check your file path and the Excel sheet.")

# Handle missing values using SimpleImputer with median strategy
imputer = SimpleImputer(strategy='median')
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Remove outliers
Q1 = df_imputed.quantile(0.25)
Q3 = df_imputed.quantile(0.75)
IQR = Q3 - Q1
df_cleaned = df_imputed[~((df_imputed < (Q1 - 1.5 * IQR)) | (df_imputed > (Q3 + 1.5 * IQR))).any(axis=1)]

# Calculate and print correlation matrix
correlation_matrix = df_cleaned.corr()
print("Correlation matrix:\n", correlation_matrix)

# Select features based on correlation to 'CI'
target_correlation = correlation_matrix['CI'].abs().sort_values(ascending=False)
selected_features = target_correlation[target_correlation > 0.05].index.tolist()
if 'CI' in selected_features:
    selected_features.remove('CI')
print("Selected features based on correlation with target:", selected_features)

# Prepare the data
X = df_cleaned[selected_features]
y = df_cleaned['CI']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define hyperparameters for Random Forest, Gradient Boosting, and KNN
hyperparameters = {
    "Random Forest": {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    "Gradient Boosting": {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 10]
    },
    "KNN": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    }
}

# Function to optimize and evaluate models
def optimize_and_evaluate(model, params, X_train, X_test, y_train, y_test):
    random_search = RandomizedSearchCV(model, params, n_iter=20, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1)
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    
    predictions = best_model.predict(X_test)
    r2 = r2_score(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    mae = mean_absolute_error(y_test, predictions)
    return r2, rmse, mae, random_search.best_params_

# Evaluate selected models
results = {}
models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "KNN": KNeighborsRegressor()
}

for name, model_instance in models.items():
    print(f"Evaluating {name}...")
    r2, rmse, mae, best_params = optimize_and_evaluate(model_instance, hyperparameters[name], X_train_scaled, X_test_scaled, y_train, y_test)
    results[name] = {"R2": r2, "RMSE": rmse, "MAE": mae, "Best Params": best_params}

# Display results
for model_name, metrics in results.items():
    print(f"{model_name} - R2: {metrics['R2']}, RMSE: {metrics['RMSE']}, MAE: {metrics['MAE']}, Best Params: {metrics['Best Params']}")
