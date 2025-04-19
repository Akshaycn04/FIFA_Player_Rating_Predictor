import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor
import time
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold

# Set plot style
plt.style.use('ggplot')
sns.set_palette("viridis")

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

# Select relevant features
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

# ...rest of the code remains the same...
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

# Improved Binning Regressor
class BinningRegressor:
    def __init__(self, bin_size):
        self.bin_size = bin_size
        self.bin_means = {}
        self.centroids = None
        self.feature_weights = None
        
    def fit(self, X, y):
        # Calculate feature weights based on correlation with the target
        feature_correlations = np.abs(np.array([np.corrcoef(X[:, i], y)[0, 1] for i in range(X.shape[1])]))
        self.feature_weights = feature_correlations / np.sum(feature_correlations)
        
        # Apply weights to features
        X_weighted = X * self.feature_weights
        
        # Create bins based on weighted feature sums
        weighted_sums = np.sum(X_weighted, axis=1)
        sorted_indices = np.argsort(weighted_sums)
        bin_size = len(y) // self.bin_size
        
        # Store bin boundaries and means
        self.bin_boundaries = []
        for i in range(self.bin_size):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < self.bin_size - 1 else len(y)
            bin_indices = sorted_indices[start_idx:end_idx]
            
            # Store bin information only if there are samples in the bin
            if len(bin_indices) > 0:
                if i > 0:  # Not the first bin
                    self.bin_boundaries.append(weighted_sums[sorted_indices[start_idx]])
                if i == self.bin_size - 1:  # Last bin
                    self.bin_boundaries.append(float('inf'))
                self.bin_means[i] = np.mean(y.iloc[bin_indices])
                
                # Also store the centroid of this bin for better prediction
                if i == 0:  # First bin
                    self.bin_boundaries.insert(0, float('-inf'))
                    
        return self
    
    def predict(self, X):
        # Apply same feature weighting
        X_weighted = X * self.feature_weights
        weighted_sums = np.sum(X_weighted, axis=1)
        predictions = []
        
        for sum_val in weighted_sums:
            # Find which bin this sample falls into
            for i in range(len(self.bin_boundaries) - 1):
                if self.bin_boundaries[i] <= sum_val < self.bin_boundaries[i+1]:
                    predictions.append(self.bin_means.get(i, np.mean(list(self.bin_means.values()))))
                    break
            else:
                # If it doesn't fall into any bin (shouldn't happen), use the overall mean
                predictions.append(np.mean(list(self.bin_means.values())))
                
        return np.array(predictions)

print("\nTraining models...")

bin_sizes = [5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 100]
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

# Test Binning Regressor
for bin_size in bin_sizes:
    fold_accuracies = []
    
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model = BinningRegressor(bin_size)
        start_train = time.time()
        model.fit(X_fold_train, y_fold_train)
        train_time = time.time() - start_train
        
        start_test = time.time()
        y_pred = model.predict(X_fold_val)
        test_time = time.time() - start_test
        
        eval_results = evaluate_model(y_fold_val, y_pred, train_time, test_time)
        fold_accuracies.append(eval_results['accuracy'])
    
    avg_accuracy = np.mean(fold_accuracies)
    
    model_final = BinningRegressor(bin_size)
    start_train_final = time.time()
    model_final.fit(X_train_scaled, y_train)
    train_time_final = time.time() - start_train_final
    
    start_test_final = time.time()
    y_pred_final = model_final.predict(X_test_scaled)
    test_time_final = time.time() - start_test_final
    
    eval_results = evaluate_model(y_test, y_pred_final, train_time_final, test_time_final)
    eval_results['model_type'] = 'Binning'
    eval_results['param'] = bin_size
    results.append(eval_results)

results_df = pd.DataFrame(results)
print("\nModel evaluation complete.")

best_model_idx = results_df['accuracy'].idxmax()
best_model = results_df.iloc[best_model_idx]

# Find best bin size details
binning_results = results_df[results_df['model_type'] == 'Binning']
best_bin_model = binning_results.loc[binning_results['accuracy'].idxmax()]

print("\nBest Binning Model Details:")
print(f"Bin size: {best_bin_model['param']}")
print(f"R² Score: {best_bin_model['r2']:.4f}")
print(f"RMSE: {best_bin_model['rmse']:.4f}")
print(f"MAE: {best_bin_model['mae']:.4f}")
print(f"Accuracy: {best_bin_model['accuracy']:.2f}%")
print(f"Training time: {best_bin_model['train_time']:.4f} seconds")
print(f"Testing time: {best_bin_model['test_time']:.4f} seconds")

summary_df = results_df.sort_values('accuracy', ascending=False).head(10)

# Visualization 1
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(binning_results['param'], binning_results['rmse'], marker='o', label='Binning')
plt.xlabel('Bin Size')
plt.ylabel('RMSE (lower is better)')
plt.title('RMSE by Bin Size')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(binning_results['param'], binning_results['accuracy'], marker='o', label='Binning')
plt.xlabel('Bin Size')
plt.ylabel('Accuracy (%)')
plt.title('Accuracy by Bin Size')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(binning_results['param'], binning_results['train_time'], marker='o', label='Binning')
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
ax.set_xticklabels([f"Binning({row['param']})" for _, row in summary_df.iterrows()], rotation=45)
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
plt.title(f'Actual vs Predicted Ratings\n{best_model["model_type"]} with parameter={best_model["param"]}')
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
