import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import plot_tree

diabetes = load_diabetes()

X, y = diabetes.data, diabetes.target

feature_names = diabetes.feature_names

print("Forma di X:", X.shape)
print("Forma di y:", y.shape)
print("Caratteristiche:", feature_names)

rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

rf_regressor.fit(X, y)

importances = rf_regressor.feature_importances_

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))

plt.title("Feature Importance")

plt.bar(
    range(X.shape[1]),
    importances[indices],
    align="center"
)

plt.xticks(
    range(X.shape[1]),
    np.array(feature_names)[indices],
    rotation=45,
    ha="right"
)

plt.ylabel("Relative Feature Importance")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("feature_importance_diabetes.png", dpi=300)
plt.show()

print("\nImportanza delle Feature")

for i in range(5):
    print(
        f"{i + 1} - {feature_names[indices[i]]}: "
        f"{importances[indices[i]]:.3f}"
    )

tree_num = -1

max_depth = 3

estimator = rf_regressor.estimators_[tree_num]

print("Totale alberi:", len(rf_regressor.estimators_))

plt.figure(figsize=(20, 10))

plot_tree(
    estimator,
    feature_names=feature_names,
    max_depth=max_depth,
    filled=True,
    fontsize=10
)

plt.title("Singolo albero della Random Forest - Diabetes")
plt.tight_layout()
plt.savefig("single_tree_diabetes.png", dpi=300)
plt.show()