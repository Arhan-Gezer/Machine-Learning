import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEATURE_NAMES = ["Neighborhood", "Price (TRY)", "Age (Years)", "Net Square Meters (m2)"]
FEATURE_TYPES = ["categorical", "numerical", "numerical", "numerical"]
CLASS_LABELS = ["low", "medium", "high", "very high"]

#Part A
# node structure for decision tree
class Node:
    def __init__(
        self,
        feature=None,
        split_value=None,
        is_categorical=False,
        left=None,
        right=None,
        value=None,
        majority=None
    ):
        self.feature = feature
        self.split_value = split_value
        self.is_categorical = is_categorical
        self.left = left
        self.right = right
        self.value = value
        self.majority = majority

def class_counts(y):   #value = [4,2,3,4]
    counts = [
        int(np.sum(y == "low")),
        int(np.sum(y == "medium")),
        int(np.sum(y == "high")),
        int(np.sum(y == "very high"))
    ]
    return counts
#print(class_counts(df_train))

#return class that appears most in the node
def majority_class(y):
    classes, counts = np.unique(y, return_counts=True)
    return classes[np.argmax(counts)]

#print(majority_class(df_train['Sınıf']))

# check if all samples in the node belong to the same class
def is_pure(y):
    return len(np.unique(y)) == 1

def gini(y):  #y = 'Sınıf' # lower impurity better
    classes = np.unique(y)
    impurity = 1.0
    for c in classes:
        probability = np.sum(y==c) / len(y)
        impurity -= probability ** 2
    return impurity

#split dataset into left and right according to split value
def split(x,y,feature_index, split_value, is_categorical):
    if is_categorical:
        left_mask = x[:,feature_index] == split_value
        right_mask = x[:,feature_index] != split_value
    else:
        left_mask = x[:, feature_index] <= split_value
        right_mask = x[:, feature_index] > split_value

    x_left = x[left_mask]
    y_left = y[left_mask]
    x_right = x[right_mask]
    y_right = y[right_mask]
    return x_left, y_left, x_right, y_right

#try all possible splits and choose the one with minimum weighted gini impurity
def best_split(x,y,feature_types): #x_train,y_train   # feature types is added because random forest may misunderstood the order
    best_gini = 1
    best_feature = None
    best_split_value = None
    best_is_categorical = False

    num_o_samples, num_o_features= x.shape

    for feature_index in range(num_o_features):
        unique_values = np.unique(x[:, feature_index])
        is_categorical = feature_types[feature_index] == "categorical"


        for split_value in unique_values:
            x_left, y_left, x_right, y_right =split(x,y,feature_index, split_value, is_categorical)

            if len(y_left)==0 or len(y_right) == 0:
                continue

            left_gini = gini(y_left)
            right_gini = gini(y_right)

            weighted_gini = (len(y_left)/len(y))*left_gini + (len(y_right)/len(y))*right_gini
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_split_value = split_value
                best_feature = feature_index
                best_is_categorical = is_categorical
    return best_gini, best_feature, best_split_value, best_is_categorical

#recursive tree build till node is pure
def tree(x,y,feature_types):
    if is_pure(y):
        return Node(value=class_counts(y), majority=y[0])

    best_gini, best_feature, best_split_value, best_is_categorical = best_split(x, y,feature_types)

    # if no valid split is found, create a leaf with the majority class
    if best_feature is None:
        return Node(value=class_counts(y), majority=majority_class(y))

    x_left, y_left, x_right, y_right = split(x, y, best_feature, best_split_value,best_is_categorical)
    left_child = tree(x_left, y_left,feature_types)
    right_child = tree(x_right, y_right,feature_types)
    return Node(
        feature=best_feature,
        split_value=best_split_value,
        is_categorical=best_is_categorical,
        left=left_child,
        right=right_child,
        value=class_counts(y),
        majority=majority_class(y)
    )

#predict class for single sample by using it recursively
def predict_one(x, node):   #one row

    if node.left is None and node.right is None:
        return node.majority

    if node.is_categorical:
        if x[node.feature] == node.split_value:
            return predict_one(x, node.left)
        else:
         return predict_one(x, node.right)
    else:
        if x[node.feature] <= node.split_value:
            return  predict_one(x,node.left)
        else:
            return predict_one(x,node.right)

def predict(x, root):   #all rows
    predictions = []
    for row in x:
        predictions.append(predict_one(row, root))
    return np.array(predictions)

#accuracy = number of correct predictions / total number of samples
def accuracy(y_true, y_pred):
    true_guess = np.sum(y_true == y_pred)
    return true_guess / len(y_true)

#create text to show inside each tree node while plotting
def node_label(node):
    if node.left is None and node.right is None:
        return f"leaf node\nvalue = {node.value}\nsınıf = {node.majority}"
    feature_name = FEATURE_NAMES[node.feature]   #Part B lazım değil sabit ondan

    if node.is_categorical:
        condition = f"{feature_name} == {node.split_value}"
    else:
        condition = f"{feature_name} <= {node.split_value}"
    return f"{condition}\nvalue = {node.value}\nsınıf = {node.majority}"

#drawing the fully grown decision tree recursively
def plot_tree(node, x=0.5, y=1.0, dx=0.35, dy=0.10, plot_area=None):  #dx=0.35, dy=0.10 seems better for jpg
    if plot_area is None:
        fig, plot_area = plt.subplots(figsize=(50, 25))
        plot_area.set_xlim(0, 1)
        plot_area.set_ylim(0, 1)
        plot_area.axis("off")

    plot_area.text(
        x, y, node_label(node),
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.3", fc="lightblue", ec="black")
    )

    if node.left is not None:
        x_left = x - dx
        y_left = y - dy

        # draw line between parent and left child
        plot_area.plot([x, x_left], [y - 0.02, y_left + 0.02], "k-")
        # write True on the branch
        plot_area.text((x + x_left) / 2, (y + y_left) / 2, "True", ha="center")
        #continue recursively
        plot_tree(node.left, x_left, y_left, dx / 2, dy, plot_area)

    if node.right is not None:
        x_right = x + dx
        y_right = y - dy

        plot_area.plot([x, x_right], [y - 0.02, y_right + 0.02], "k-")
        plot_area.text((x + x_right) / 2, (y + y_right) / 2, "False", ha="center")

        plot_tree(node.right, x_right, y_right, dx / 2, dy, plot_area)

    return plot_area

#calculate tp,tn,fp,fn for one class
def confusion_matrix_per_class(y_true, y_pred, class_label):
    tp = np.sum((y_true == class_label) & (y_pred == class_label))
    tn = np.sum((y_true != class_label) & (y_pred != class_label))
    fp = np.sum((y_true != class_label) & (y_pred == class_label))
    fn = np.sum((y_true == class_label) & (y_pred != class_label))
    return tp,tn,fp,fn

# calculate performance metrics
def results(y_true, y_pred):
    accuracy_result = accuracy(y_true, y_pred)
#TP Rate = TP / (TP + FN)
#TN Rate = TN / (TN + FP)
#Precision = TP / (TP + FP)
#F Score = 2 * (Precision * Recall) / (Precision + Recall)
    tp_rate_list = []
    tn_rate_list = []
    precision_list = []
    f_score_list = []

    total_tp_num = 0
    total_tn_num = 0

    for class_label in CLASS_LABELS:
        tp,tn,fp,fn = confusion_matrix_per_class(y_true,y_pred, class_label)
        total_tp_num += tp
        total_tn_num += tn

        if (tp + fn) > 0:
            tp_rate_value = tp / (tp +fn)
        else:
            tp_rate_value = 0
        if (tn + fp) > 0:
            tn_rate_value = tn / (tn + fp)
        else:
            tn_rate_value = 0
        if (tp + fp) > 0:
            precision_value = tp / (tp + fp)
        else:
            precision_value = 0
        if (precision_value + tp_rate_value) > 0:
            f_score_value = 2 * precision_value * tp_rate_value / (precision_value + tp_rate_value)
        else: f_score_value = 0

        tp_rate_list.append(tp_rate_value)
        tn_rate_list.append(tn_rate_value)
        precision_list.append(precision_value)
        f_score_list.append(f_score_value)

    avg_tp_rate = np.mean(tp_rate_list)
    avg_tn_rate = np.mean(tn_rate_list)
    avg_precision = np.mean(precision_list)
    avg_f_score = np.mean(f_score_list)
    return accuracy_result, avg_tp_rate, avg_tn_rate, avg_precision, avg_f_score, total_tp_num, total_tn_num

def print_results(title, results):
    accuracy, tp_rate, tn_rate, precision, f_score, total_tp, total_tn = results
    print(title)
    print(f"Accuracy: {accuracy:.3f}")
    print(f"TP Rate (Recall): {tp_rate:.3f}")
    print(f"TN Rate: {tn_rate:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"F-Score: {f_score:.3f}")
    print(f"Total Number of TP: {total_tp}")
    print(f"Total Number of TN: {total_tn}")

#Part B
#Random Forest using random subspace method
#each tree is trained with randomly selected feature subsets

# return the most frequently predicted class among trees
def most_repeated_class(predictions):
    classes, counts = np.unique(predictions, return_counts=True)
    return classes[np.argmax(counts)]

# train multiple decision trees
def random_forest_train(x_train, y_train, num_of_trees, feature_size):
    forest = []
    num_of_features = 4
    for i in range(num_of_trees):
        feature_choice = np.random.choice(num_of_features, feature_size, replace=False)  # randomly choose features
        x_new = x_train[:,feature_choice]
        new_feature_types = [FEATURE_TYPES[j] for j in feature_choice]
        root = tree(x_new,y_train, new_feature_types)
        forest.append((root,feature_choice))
    return forest

# get predictions from all trees and use most repeated class for final result
def random_forest_predict(x, forest):
    final_predictions = []
    for row in x:
        tree_predictions = []
        for root, feature_choice in forest:
            new_row = row[feature_choice]
            prediction = predict_one(new_row, root)
            tree_predictions.append(prediction)
        final_predictions.append(most_repeated_class(tree_predictions))
    return  np.array(final_predictions)

df_train = pd.read_excel("X_train.xlsx")
df_test = pd.read_excel("X_test.xlsx")

#dedebaşı is different in two dataframes
df_train["Neighborhood"] = df_train["Neighborhood"].str.strip()
df_test["Neighborhood"] = df_test["Neighborhood"].str.strip()

x_train = df_train.drop("Sınıf", axis=1).values
y_train = df_train["Sınıf"].values

x_test = df_test.drop("Sınıf", axis=1).values
y_test = df_test["Sınıf"].values

#Part A
root = tree(x_train, y_train,FEATURE_TYPES)
#print("root.feature:", root.feature)
#print("root.split_value:", root.split_value)
#print("root.is_categorical:", root.is_categorical)

train_predictions = predict(x_train, root)
test_predictions = predict(x_test, root)

#train_accuracy = accuracy(y_train, train_predictions)
#test_accuracy = accuracy(y_test, test_predictions)

#print(x_train[:5])
#print(df_train['Neighborhood'].dtype)
train_results = results(y_train, train_predictions)
test_results = results(y_test, test_predictions)

print_results("Train Results:", train_results)
print_results("Test Results:", test_results)

#Part B    # 2 düşük feature size olduğu için accuracy düşük
forest = random_forest_train(x_train, y_train,num_of_trees=15, feature_size=2)

rf_test_predictions = random_forest_predict(x_test, forest)
rf_test_results = results(y_test, rf_test_predictions)

print_results("Random Forest Test Results:", rf_test_results)

#Part A drawing
ax = plot_tree(root)

plt.savefig("decision_tree.jpg", bbox_inches="tight")
plt.show()




