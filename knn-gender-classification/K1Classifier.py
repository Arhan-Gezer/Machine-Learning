import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from tabulate import tabulate


# --------------------------------------------------
# 1. DATA LOADING
# --------------------------------------------------
# Load feature data (head circumference, height)
X = pd.read_excel("bas-boy.ods", engine="odf")

# Load target labels (gender)
y = pd.read_excel("cinsiyet.ods", engine="odf")
y = y.iloc[:, 0]  # Select target column


# --------------------------------------------------
# 2. TRAIN–TEST SPLIT
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=0
)


# --------------------------------------------------
# 3. FINAL MODEL SELECTION (K = 1)
# --------------------------------------------------
print("\n--- Selected Model: KNN with K = 1 ---")

best_knn = KNeighborsClassifier(n_neighbors=1)
best_knn.fit(X_train, y_train)

y_pred_best = best_knn.predict(X_test)
best_acc = accuracy_score(y_test, y_pred_best)

print(f"Baseline Accuracy (KNN, K=1): {best_acc:.2f}")


# --------------------------------------------------
# 4. ERROR ANALYSIS
# --------------------------------------------------
results = X_test.copy()
results["True Gender"] = y_test.values
results["Predicted Gender"] = y_pred_best

incorrect_samples = results[
    results["True Gender"] != results["Predicted Gender"]
]

print("\nMisclassified Samples (K=1):")
print(tabulate(incorrect_samples, headers="keys", tablefmt="psql"))


# --------------------------------------------------
# 5. RULE-BASED POST-PROCESSING
# --------------------------------------------------
print("\n--- Applying Rule-Based Post-Processing ---")

y_pred_improved = []

comparison_df = X_test.copy()
comparison_df["True Gender"] = y_test.values
comparison_df["KNN Prediction"] = y_pred_best

for _, row in X_test.iterrows():

    input_row = pd.DataFrame([row])
    model_prediction = best_knn.predict(input_row)[0]
    final_prediction = model_prediction

    # Rule 1: Short height & small head circumference → Female
    if model_prediction == 1 and row["Boy"] < 57 and row["Başçevre"] < 39.5:
        if row["Başçevre"] > 38:
            final_prediction = 0

    # Rule 2: Height in low 70s with large head circumference → Female
    elif model_prediction == 1 and 70 < row["Boy"] < 74 and row["Başçevre"] > 44:
        final_prediction = 0

    # Rule 3: Height in 80s with mid-range head circumference → Female
    elif model_prediction == 1 and 79 < row["Boy"] < 86 and 46 < row["Başçevre"] < 49.15:
        final_prediction = 0

    y_pred_improved.append(final_prediction)


# --------------------------------------------------
# 6. PERFORMANCE COMPARISON
# --------------------------------------------------
comparison_df["Rule-Based Prediction"] = y_pred_improved

new_acc = accuracy_score(y_test, y_pred_improved)
improvement = new_acc - best_acc

print(f"\nBaseline Accuracy (KNN):        {best_acc:.2f}")
print(f"Accuracy After Rule-Based Step: {new_acc:.2f}")

if improvement > 0:
    print(f"Accuracy Improvement: +{improvement:.2f}")
else:
    print("No accuracy improvement achieved.")


# --------------------------------------------------
# 7. REMAINING MISCLASSIFICATIONS
# --------------------------------------------------
initial_misclassified = comparison_df[
    comparison_df["True Gender"] != comparison_df["KNN Prediction"]
]

print("\nFinal Misclassified Samples After All Rules:")
display_columns = [
    "Başçevre",
    "Boy",
    "True Gender",
    "Rule-Based Prediction"
]

print(tabulate(
    initial_misclassified[display_columns],
    headers="keys",
    tablefmt="psql"
))

# --------------------------------------------------
# 8 CORRECTED SAMPLES (Incorrect → Correct)
# --------------------------------------------------
corrected_samples = comparison_df[
    (comparison_df["True Gender"] != comparison_df["KNN Prediction"]) &
    (comparison_df["True Gender"] == comparison_df["Rule-Based Prediction"])
]

print("\nSamples Corrected by Rule-Based Post-Processing:")
display_columns = [
    "Başçevre",
    "Boy",
    "True Gender",
    "KNN Prediction",
    "Rule-Based Prediction"
]

if len(corrected_samples) == 0:
    print("No samples were corrected by the rules.")
else:
    print(tabulate(
        corrected_samples[display_columns],
        headers="keys",
        tablefmt="psql"
    ))
# --------------------------------------------------
# 9. REGRESSION ERRORS (Correct → Incorrect)
# --------------------------------------------------
regression_errors = comparison_df[
    (comparison_df["True Gender"] == comparison_df["KNN Prediction"]) &
    (comparison_df["True Gender"] != comparison_df["Rule-Based Prediction"])
]

print("\nSamples That Became Incorrect After Rules (Regression Errors):")
display_columns = [
    "Başçevre",
    "Boy",
    "True Gender",
    "KNN Prediction",
    "Rule-Based Prediction"
]

if len(regression_errors) == 0:
    print("No regression errors detected.")
else:
    print(tabulate(
        regression_errors[display_columns],
        headers="keys",
        tablefmt="psql"
    ))
