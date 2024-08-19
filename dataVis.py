import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
import numpy as np
import os

# Load the dataset
df = pd.read_csv('dataset2.csv')

# Ensure the output directory exists
output_dir = "health_status_analysis"
os.makedirs(output_dir, exist_ok=True)

# Descriptive statistics
stats = df.groupby('health_status').describe()
stats.to_csv(os.path.join(output_dir, 'descriptive_statistics.csv'))

# Visualization: Box Plots
features = df.columns[:-1]  # all columns except 'health_status'
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='health_status', y=feature, data=df)
    plt.title(f'{feature} vs Health Status (Box Plot)')
    plt.savefig(os.path.join(output_dir, f'{feature}_boxplot.png'))
    plt.close()

# Visualization: Violin Plots
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='health_status', y=feature, data=df)
    plt.title(f'{feature} vs Health Status (Violin Plot)')
    plt.savefig(os.path.join(output_dir, f'{feature}_violinplot.png'))
    plt.close()

# Correlation Matrix (Excluding 'health_status')
correlation_matrix = df[features].corr()  # Exclude 'health_status' which is categorical
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# Line Charts for Correlation Matrix
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.plot(correlation_matrix.index, correlation_matrix[feature], marker='o', label=feature)
plt.title('Correlation Line Charts')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.legend(loc='upper right')
plt.xticks(rotation=90)
plt.savefig(os.path.join(output_dir, 'correlation_line_charts.png'))
plt.close()

# Feature Importance using RandomForestClassifier
X = df[features]
y = df['health_status']

# Convert categorical labels to numerical
y = pd.factorize(y)[0]

# Train a RandomForest Classifier
clf = RandomForestClassifier(random_state=0)
clf.fit(X, y)

# Plot Feature Importance
importance = clf.feature_importances_
indices = np.argsort(importance)[::-1]

plt.figure(figsize=(10, 6))
sns.barplot(x=importance[indices], y=[features[i] for i in indices])
plt.title('Feature Importance')
plt.savefig(os.path.join(output_dir, 'feature_importance.png'))
plt.close()

# Decision Tree Visualization
plt.figure(figsize=(20, 10))
plot_tree(clf.estimators_[0], feature_names=features, class_names=['healthy', 'unhealthy', 'warning'], filled=True)
plt.title('Decision Tree')
plt.savefig(os.path.join(output_dir, 'decision_tree.png'))
plt.close()

# Random Forest Structure Visualization
plt.figure(figsize=(20, 20))
for i, tree in enumerate(clf.estimators_[:5]):  # Plot the first 5 trees for demonstration
    plt.subplot(3, 2, i + 1)
    plot_tree(tree, feature_names=features, class_names=['healthy', 'unhealthy', 'warning'], filled=True)
    plt.title(f'Tree {i+1}')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'random_forest_structure.png'))
plt.close()

print(f"Analysis complete. Plots and statistics have been saved in the '{output_dir}' directory.")

