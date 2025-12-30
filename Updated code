import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_recall_curve, 
                             average_precision_score)

# 1. DATA INGESTION & EXPLORATION
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target
target_names = iris.target_names

# Enhanced Pairplot with KDE to see feature overlaps
sns.set_theme(style="ticks", palette="pastel")
df_plot = X.copy()
df_plot['species'] = [target_names[i] for i in y]
sns.pairplot(df_plot, hue='species', diag_kind="kde", markers=["o", "s", "D"])
plt.suptitle("Feature Distribution and Species Separation", y=1.02)
plt.show()

# 2. DATA PREPROCESSING PIPELINE
# Stratified split ensures class balance in both training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Feature Scaling: Crucial for KNN because it uses distance metrics (Euclidean)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. HYPERPARAMETER OPTIMIZATION (Grid Search)
# We test 120 combinations to find the best k and distance formula
param_grid = {
    'n_neighbors': np.arange(1, 31),
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_knn = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# 4. MODEL EVALUATION
y_pred = best_knn.predict(X_test_scaled)
y_score = best_knn.predict_proba(X_test_scaled) # Probabilities for PR Curve

# 5. DIAGNOSTIC VISUALIZATIONS
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# A. Confusion Matrix Heatmap
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='mako', ax=ax[0],
            xticklabels=target_names, yticklabels=target_names)
ax[0].set_title('Confusion Matrix: Prediction Accuracy')
ax[0].set_xlabel('Predicted Species')
ax[0].set_ylabel('Actual Species')

# B. Precision-Recall Curve (Multi-class)
y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
for i in range(len(target_names)):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    ax[1].plot(recall, precision, lw=2, label=f'Class {target_names[i]}')

ax[1].set_title('Precision-Recall Curves')
ax[1].set_xlabel('Recall')
ax[1].set_ylabel('Precision')
ax[1].legend(loc="best")
plt.show()

# 6. FINAL PERFORMANCE REPORT
print("\n" + "="*30)
print(f"TEST ACCURACY: {accuracy_score(y_test, y_pred):.2%}")
print("="*30)
print(classification_report(y_test, y_pred, target_names=target_names))
