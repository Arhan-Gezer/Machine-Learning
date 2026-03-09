import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tabulate import tabulate


# Load feature data (head circumference, height, etc.)
X = pd.read_excel("bas-boy.ods", engine="odf")

# Load target labels (gender)
y = pd.read_excel("cinsiyet.ods", engine="odf")
y = y.iloc[:, 0]

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)

# Try different K values from 1 to 15
for k in range(1, 16):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"K = {k} -> Test Accuracy = {acc:.2f}")

# Analyze misclassified samples
results = X_test.copy()
results["True Gender"] = y_test.values
results["Prediction"] = y_pred

wrong_predictions = results[results["True Gender"] != results["Prediction"]]

print("\nMisclassified Samples:")
# tablefmt='psql' creates a clean table format
print(tabulate(wrong_predictions, headers='keys', tablefmt='psql'))
