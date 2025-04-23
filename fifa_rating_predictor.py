# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
import os
# Ignore warnings for clean output
warnings.filterwarnings('ignore')

# Import Scikit-learn modules
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.base import BaseEstimator, RegressorMixin

# Set plot style
plt.style.use('ggplot')
sns.set_palette("viridis")

# Define custom smoother model using HistGradientBoostingRegressor
class SmootherRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, bin_width):
        self.bin_width = bin_width
        self.model = HistGradientBoostingRegressor(max_bins=255, random_state=42)

    def fit(self, X, y):
        # Adjust features to match bin_width
        self.X_min = X.min(axis=0)
        self.X_max = X.max(axis=0)
        self.n_bins = np.ceil((self.X_max - self.X_min) / self.bin_width).astype(int).clip(min=2, max=255)
        self.model.set_params(max_bins=self.n_bins.max())
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

# Start the process
print("FIFA Player Rating Predictor - Non-Parametric Regression (Using Bin Width)")
print("-" * 60)

# Load dataset
print("Loading dataset...")
import kagglehub
path = kagglehub.dataset_download("maso0dahmed/football-players-data")
csv_file = [f for f in os.listdir(path) if f.endswith(".csv")][0]
df = pd.read_csv(os.path.join(path,csv_file))
print("Path to dataset files:", path)

# Remove goalkeepers (GK) as they have very different stats
df = df[~df['positions'].str.startswith('GK')]

# Basic dataset information
print(f"\nDataset shape: {df.shape}")
print(f"Number of players: {len(df)}")
print(f"Rating range: {df['overall_rating'].min()} - {df['overall_rating'].max()}")

# Select relevant features for modeling
relevant_features = [
    'finishing', 'ball_control', 'dribbling', 'curve', 'freekick_accuracy',
    'long_passing', 'short_passing', 'volleys', 'crossing',
    'sprint_speed', 'acceleration', 'stamina', 'strength', 'jumping',
    'agility', 'balance', 'reactions','international_reputation(1-5)', 'weak_foot(1-5)','skill_moves(1-5)',
    'vision', 'composure', 'penalties', 'positioning', 'interceptions',
    'aggression', 'marking', 'standing_tackle', 'sliding_tackle',
    'shot_power', 'long_shots', 'heading_accuracy',
    'age','height_cm', 'weight_kgs'
]

# Keep only available features present in the dataset
available_features = [f for f in relevant_features if f in df.columns]
print(f"\nUsing {len(available_features)} relevant features:")
print(available_features)

# Define features (X) and target variable (y)
X = df[available_features]
y = df['overall_rating']

# Remove any rows with missing values
X = X.dropna()
y = y[X.index]

print(f"\nFinal dataset shape after cleaning: {X.shape}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining models...")

# List of bin widths to test
bin_widths = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Container to store results
results = []

# Function to evaluate model performance
def evaluate_model(y_true, y_pred, train_time, test_time):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    accuracy = 100 * (1 - np.mean(np.abs(y_true - y_pred) / y_true))
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'train_time': train_time,
        'test_time': test_time,
        'predictions': y_pred
    }

# 5-fold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Loop through different bin widths
for bin_width in bin_widths:
    fold_accuracies = []

    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        model = SmootherRegressor(bin_width)
        start_train = time.time()
        model.fit(X_fold_train, y_fold_train)
        train_time = time.time() - start_train

        start_test = time.time()
        y_pred = model.predict(X_fold_val)
        test_time = time.time() - start_test

        eval_results = evaluate_model(y_fold_val, y_pred, train_time, test_time)
        fold_accuracies.append(eval_results['accuracy'])

    avg_accuracy = np.mean(fold_accuracies)

    # Train on full training set
    model_final = SmootherRegressor(bin_width)
    start_train_final = time.time()
    model_final.fit(X_train_scaled, y_train)
    train_time_final = time.time() - start_train_final

    start_test_final = time.time()
    y_pred_final = model_final.predict(X_test_scaled)
    test_time_final = time.time() - start_test_final

    eval_results = evaluate_model(y_test, y_pred_final, train_time_final, test_time_final)
    eval_results['model_type'] = 'Smoothed'
    eval_results['param'] = bin_width
    results.append(eval_results)

# Create a DataFrame with all results
results_df = pd.DataFrame(results)
print("\nModel evaluation complete.")

# Find the best model (highest accuracy)
best_model_idx = results_df['accuracy'].idxmax()
best_model = results_df.iloc[best_model_idx]

# Display best model performance
print("\nBest Smoother Model Details:")
print(f"Bin Width: {best_model['param']}")
print(f"R² Score: {best_model['r2']:.4f}")
print(f"RMSE: {best_model['rmse']:.4f}")
print(f"MAE: {best_model['mae']:.4f}")
print(f"Accuracy: {best_model['accuracy']:.2f}%")
print(f"Training time: {best_model['train_time']:.4f} seconds")
print(f"Testing time: {best_model['test_time']:.4f} seconds")

# Show the top 10 models
summary_df = results_df.sort_values('accuracy', ascending=False).head(10)

# -----------------------------------
# Visualization 1: Performance Metrics
# -----------------------------------
plt.figure(figsize=(15, 10))

# RMSE by Bin Width
plt.subplot(2, 2, 1)
plt.plot(results_df['param'], results_df['rmse'], marker='o')
plt.xlabel('Bin Width')
plt.ylabel('RMSE (lower is better)')
plt.title('RMSE by Bin Width')
plt.grid(True, alpha=0.3)

# Accuracy by Bin Width
plt.subplot(2, 2, 2)
plt.plot(results_df['param'], results_df['accuracy'], marker='o')
plt.xlabel('Bin Width')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy by Bin Width')
plt.grid(True, alpha=0.3)

# Training Time by Bin Width
plt.subplot(2, 2, 3)
plt.plot(results_df['param'], results_df['train_time'], marker='o')
plt.xlabel('Bin Width')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time by Bin Width')
plt.grid(True, alpha=0.3)

# Top 10 models: Accuracy vs RMSE
plt.subplot(2, 2, 4)
ax = plt.gca()
bar_width = 0.35
index = np.arange(len(summary_df))
bar1 = ax.bar(index, summary_df['accuracy'], bar_width, label='Accuracy (%)')
ax2 = ax.twinx()
bar2 = ax2.bar(index + bar_width, summary_df['rmse'], bar_width, color='lightcoral', label='RMSE')
ax.set_xlabel('Models')
ax.set_ylabel('Accuracy (%)')
ax2.set_ylabel('RMSE')
ax.set_title('Top 10 Models: Accuracy vs RMSE')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels([f"Width({row['param']})" for _, row in summary_df.iterrows()], rotation=45)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()

# -----------------------------------
# Visualization 2: Predictions vs Actual
# -----------------------------------
plt.figure(figsize=(15, 10))

best_predictions = best_model['predictions']
errors = np.abs(y_test - best_predictions)

error_df = pd.DataFrame({
    'Player': df.loc[y_test.index, 'name'],
    'Actual': y_test,
    'Predicted': best_predictions,
    'Error': errors
})

# Actual vs Predicted scatter plot
plt.subplot(2, 2, 1)
plt.scatter(y_test, best_predictions, alpha=0.5, c=errors, cmap='coolwarm')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.colorbar(label='Absolute Error')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title(f'Actual vs Predicted Ratings\nSmoothed with bin_width={best_model["param"]}')
plt.grid(True, alpha=0.3)

# Error distribution
plt.subplot(2, 2, 2)
sns.histplot(errors, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

# Error vs Actual Rating
plt.subplot(2, 2, 3)
plt.scatter(y_test, errors, alpha=0.5, c=errors, cmap='coolwarm')
plt.xlabel('Actual Rating')
plt.ylabel('Prediction Error')
plt.title('Error vs Actual Rating')
plt.grid(True, alpha=0.3)

# Top 5 worst predictions
plt.subplot(2, 2, 4)
worst_predictions = error_df.nlargest(5, 'Error')
sns.barplot(x='Player', y='Error', data=worst_predictions)
plt.xticks(rotation=45, ha='right')
plt.title('Top 5 Largest Prediction Errors')
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.tight_layout()
plt.show()
# -----------------------------------
# Print best/worst predictions
# -----------------------------------
print("\nTop 5 Most Accurate Predictions:")
best_predictions = error_df.nsmallest(5, 'Error')
print(best_predictions[['Player', 'Actual', 'Predicted', 'Error']].to_string(index=False))

print("\nTop 5 Least Accurate Predictions:")
worst_predictions = error_df.nlargest(5, 'Error')
print(worst_predictions[['Player', 'Actual', 'Predicted', 'Error']].to_string(index=False))



print("\nFIFA Player Rating Prediction Complete!")
