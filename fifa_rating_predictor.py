import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.base import BaseEstimator, RegressorMixin
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold

# Set plot style
plt.style.use('ggplot')
sns.set_palette("viridis")

class BinnedRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, bin_size):
        self.bin_size = bin_size
        self.binner = KBinsDiscretizer(n_bins=bin_size, encode='ordinal', strategy='kmeans')
        self.feature_weights = None
        self.bin_models = None
        
    def fit(self, X, y):
        # Convert to numpy arrays
        y_values = y.values if isinstance(y, pd.Series) else y
        
        # Calculate feature importances using correlation
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y_values)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        # Normalize feature weights
        self.feature_weights = np.array(correlations)
        self.feature_weights = self.feature_weights / np.sum(self.feature_weights)
        
        # Create weighted feature
        X_weighted = X * self.feature_weights
        projected_feature = np.sum(X_weighted, axis=1).reshape(-1, 1)
        
        # Bin the data
        bin_indices = self.binner.fit_transform(projected_feature).ravel()
        
        # Create bin models
        self.bin_models = {}
        for bin_idx in range(self.bin_size):
            mask = (bin_indices == bin_idx)
            if np.sum(mask) > 0:
                self.bin_models[bin_idx] = {
                    'mean': np.mean(y_values[mask]),
                    'std': np.std(y_values[mask]) if np.sum(mask) > 1 else 0,
                    'count': np.sum(mask)
                }
            else:
                self.bin_models[bin_idx] = None
        
        # Handle empty bins through interpolation
        empty_bins = [i for i in range(self.bin_size) if self.bin_models[i] is None]
        for empty_bin in empty_bins:
            # Find nearest non-empty bins
            left_val = right_val = None
            left_idx = empty_bin - 1
            right_idx = empty_bin + 1
            
            while left_idx >= 0:
                if self.bin_models[left_idx] is not None:
                    left_val = self.bin_models[left_idx]['mean']
                    break
                left_idx -= 1
                
            while right_idx < self.bin_size:
                if self.bin_models[right_idx] is not None:
                    right_val = self.bin_models[right_idx]['mean']
                    break
                right_idx += 1
            
            # Interpolate
            if left_val is not None and right_val is not None:
                left_dist = empty_bin - left_idx
                right_dist = right_idx - empty_bin
                total_dist = left_dist + right_dist
                mean_val = (left_val * right_dist + right_val * left_dist) / total_dist
            elif left_val is not None:
                mean_val = left_val
            elif right_val is not None:
                mean_val = right_val
            else:
                mean_val = np.mean(y_values)
                
            self.bin_models[empty_bin] = {
                'mean': mean_val,
                'std': np.std(y_values),
                'count': 0
            }
        
        # Smooth bin predictions
        means = np.array([self.bin_models[i]['mean'] for i in range(self.bin_size)])
        window = 3
        weights = np.ones(window) / window
        smoothed_means = np.convolve(means, weights, mode='same')
        
        for i in range(self.bin_size):
            self.bin_models[i]['mean'] = smoothed_means[i]
        
        return self
    
    def predict(self, X):
        # Create weighted feature
        X_weighted = X * self.feature_weights
        projected_feature = np.sum(X_weighted, axis=1).reshape(-1, 1)
        
        # Get bin assignments
        bin_indices = self.binner.transform(projected_feature).ravel()
        
        # Make predictions
        predictions = np.zeros(len(X))
        for i, bin_idx in enumerate(bin_indices):
            bin_idx = int(min(max(bin_idx, 0), self.bin_size - 1))
            predictions[i] = self.bin_models[bin_idx]['mean']
            
        return predictions
print("FIFA Player Rating Predictor - Non-Parametric Regression")
print("-" * 60)

# Load the data
print("Loading dataset...")
df = pd.read_csv('fifa_players.csv')

# Remove goalkeepers
df = df[~df['positions'].str.startswith('GK')]

# Display basic dataset information
print(f"\nDataset shape: {df.shape}")
print(f"Number of players: {len(df)}")
print(f"Rating range: {df['overall_rating'].min()} - {df['overall_rating'].max()}")

# Update relevant_features by removing 'body_type'
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

available_features = [f for f in relevant_features if f in df.columns]
print(f"\nUsing {len(available_features)} relevant features:")
print(available_features)

X = df[available_features]
y = df['overall_rating']

# Remove rows with missing values
X = X.dropna()
y = y[X.index]

print(f"\nFinal dataset shape after cleaning: {X.shape}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nTraining models...")

bin_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150]
results = []

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

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Test Binned Regressor
for bin_size in bin_sizes:
    fold_accuracies = []
    
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = BinnedRegressor(bin_size)
        start_train = time.time()
        model.fit(X_fold_train, y_fold_train)
        train_time = time.time() - start_train
        
        start_test = time.time()
        y_pred = model.predict(X_fold_val)
        test_time = time.time() - start_test
        
        eval_results = evaluate_model(y_fold_val, y_pred, train_time, test_time)
        fold_accuracies.append(eval_results['accuracy'])
    
    avg_accuracy = np.mean(fold_accuracies)
    
    model_final = BinnedRegressor(bin_size)
    start_train_final = time.time()
    model_final.fit(X_train_scaled, y_train)
    train_time_final = time.time() - start_train_final
    
    start_test_final = time.time()
    y_pred_final = model_final.predict(X_test_scaled)
    test_time_final = time.time() - start_test_final
    
    eval_results = evaluate_model(y_test, y_pred_final, train_time_final, test_time_final)
    eval_results['model_type'] = 'Binned'
    eval_results['param'] = bin_size
    results.append(eval_results)

results_df = pd.DataFrame(results)
print("\nModel evaluation complete.")

best_model_idx = results_df['accuracy'].idxmax()
best_model = results_df.iloc[best_model_idx]

# Find best bin size details
binned_results = results_df[results_df['model_type'] == 'Binned']
best_binned_model = binned_results.loc[binned_results['accuracy'].idxmax()]

print("\nBest Binned Model Details:")
print(f"Bin size: {best_binned_model['param']}")
print(f"R² Score: {best_binned_model['r2']:.4f}")
print(f"RMSE: {best_binned_model['rmse']:.4f}")
print(f"MAE: {best_binned_model['mae']:.4f}")
print(f"Accuracy: {best_binned_model['accuracy']:.2f}%")
print(f"Training time: {best_binned_model['train_time']:.4f} seconds")
print(f"Testing time: {best_binned_model['test_time']:.4f} seconds")

summary_df = results_df.sort_values('accuracy', ascending=False).head(10)

# Visualization 1
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(binned_results['param'], binned_results['rmse'], marker='o', label='Binned')
plt.xlabel('Bin Size')
plt.ylabel('RMSE (lower is better)')
plt.title('RMSE by Bin Size')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(binned_results['param'], binned_results['accuracy'], marker='o', label='Binned')
plt.xlabel('Bin Size')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy by Bin Size')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(binned_results['param'], binned_results['train_time'], marker='o', label='Binned')
plt.xlabel('Bin Size')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time by Bin Size')
plt.legend()
plt.grid(True, alpha=0.3)

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
ax.set_xticklabels([f"Binned({row['param']})" for _, row in summary_df.iterrows()], rotation=45)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()

# Visualization 2
plt.figure(figsize=(15, 10))
best_predictions = best_model['predictions']
errors = np.abs(y_test - best_predictions)

error_df = pd.DataFrame({
    'Player': df.loc[y_test.index, 'name'],
    'Actual': y_test,
    'Predicted': best_predictions,
    'Error': errors
})

plt.subplot(2, 2, 1)
plt.scatter(y_test, best_predictions, alpha=0.5, c=errors, cmap='coolwarm')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.colorbar(label='Absolute Error')
plt.xlabel('Actual Rating')
plt.ylabel('Predicted Rating')
plt.title(f'Actual vs Predicted Ratings\nBinned with bin_size={best_model["param"]}')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
sns.histplot(errors, kde=True, bins=30)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Error Distribution')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.scatter(y_test, errors, alpha=0.5, c=errors, cmap='coolwarm')
plt.xlabel('Actual Rating')
plt.ylabel('Prediction Error')
plt.title('Error vs Actual Rating')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 4)
worst_predictions = error_df.nlargest(5, 'Error')
sns.barplot(x='Player', y='Error', data=worst_predictions)
plt.xticks(rotation=45, ha='right')
plt.title('Top 5 Largest Prediction Errors')
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("\nTop 5 Most Accurate Predictions:")
best_predictions = error_df.nsmallest(5, 'Error')
print(best_predictions[['Player', 'Actual', 'Predicted', 'Error']].to_string(index=False))

print("\nTop 5 Least Accurate Predictions:")
worst_predictions = error_df.nlargest(5, 'Error')
print(worst_predictions[['Player', 'Actual', 'Predicted', 'Error']].to_string(index=False))

plt.tight_layout()
plt.show()

print("\nFIFA Player Rating Prediction Complete!")
