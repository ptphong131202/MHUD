from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
import pandas as pd

iris = pd.read_csv("soybean-update.csv")

# Define features and labels
X = iris.drop(["class"], axis=1)
y = iris["class"].astype(str)  # Convert class attribute to string

# Create decision tree with information gain criterion
clf = DecisionTreeClassifier(criterion='entropy')

# Train decision tree
clf.fit(X, y)

# Export decision tree in DOT format
dot_data = export_graphviz(clf, out_file=None,
                           feature_names=X.columns.tolist(),
                           class_names=y.unique().tolist(),  # Convert class labels to list of strings
                           filled=True, rounded=True,
                           special_characters=True)

# Draw decision tree
graph = graphviz.Source(dot_data)
graph.render('iris_decision_tree_entropy', format='png')
