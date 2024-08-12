from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

# Define Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)

# Train the model
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculate and display model statistics
n_trees = len(rf_model.estimators_)
print(f"Number of trees: {n_trees}")

# Example: Get depth of each tree
tree_depths = [estimator.tree_.max_depth for estimator in rf_model.estimators_]
average_depth = sum(tree_depths) / len(tree_depths)
print(f"Average tree depth: {average_depth}")

# Example: Get number of nodes in each tree
n_nodes = [estimator.tree_.node_count for estimator in rf_model.estimators_]
average_nodes = sum(n_nodes) / len(n_nodes)
print(f"Average number of nodes per tree: {average_nodes}")

