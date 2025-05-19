import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, balanced_accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline

# Load the dataset
name_dataset = "DatasetNoDuplicate"  # DatasetNoDuplicate
print(f"the dataset is {name_dataset}")
dataset = np.loadtxt(f"{name_dataset}.csv", delimiter=",", skiprows=1)

X = dataset[:, :-1]
y = dataset[:, -1]

# Train-Test Split
XTrain, XTest, yTrain, yTest = train_test_split(
    X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y
)
"""
# Initialize the individual models
dt_model = DecisionTreeClassifier(random_state=42)
svc_model = SVC(kernel="rbf", C=1, gamma=0.1, degree=2, random_state=42)
nb_model = GaussianNB()
logreg_model = LogisticRegression(C=1, max_iter=10000, penalty="l2", solver="lbfgs", random_state=42)
"""


"""
# Create a Voting Classifier with Hard Voting (Majority Voting)
voting_clf = VotingClassifier(
    estimators=[
        ('dt', dt_model),
        ('svc', svc_model),
        ('nb', nb_model),
        ('logreg', logreg_model)
    ],
    voting='hard'  # 'soft' for probability-based voting
)
"""

# Define Pipelines for models with appropriate scalers
dt_model = DecisionTreeClassifier()  # No scaling needed #criterion="gini", max_depth=6, max_features=9, min_samples_split=4, splitter="random",  random_state=42

svc_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # this scaler I got from the best parameter set
    ('svc', SVC(kernel="rbf", C=1, gamma="scale", degree=2, random_state=42, probability=True))
])

nb_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # GaussianNB often benefits from MinMaxScaler
    ('nb', GaussianNB())
])

logreg_pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Logistic Regression benefits from StandardScaler
    ('logreg', LogisticRegression(C=1, max_iter=10000, penalty="l2", solver="lbfgs", random_state=42))
])

# Create a Voting Classifier with Hard Voting
voting_clf = VotingClassifier(
    estimators=[
        ('dt', dt_model),
        ('svc', svc_pipeline),
        ('nb', nb_pipeline),
        ('logreg', logreg_pipeline)
    ],
    voting='hard'  # 'soft' for probability-based voting and "hard" os for the majority-based voting
)

# Train the Voting Classifier
voting_clf.fit(XTrain, yTrain)

# Predict on the test set
yPred = voting_clf.predict(XTest)
"""
# Evaluate the performance

#print(f"Accuracy Score: {accuracy_score(yTest, yPred):.4f}")
#print("\nClassification Report:")
#print(classification_report(yTest, yPred))
#print(f"Balanced Accuracy Score {balanced_accuracy_score(yTest, yPred)}")
"""
# Model Scores
tr_score = voting_clf.score(XTrain, yTrain)
test_score = voting_clf.score(XTest, yTest)
balanceaccuracy_score = balanced_accuracy_score(yTest, yPred)
mse = mean_squared_error(yTest, yPred)

print(f"Training Score: {tr_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Balance Accuracy Score: {balanceaccuracy_score:.4f}")
print(f"mean squared error: {mse:.4f}")

# Confusion Matrix
conf_matrix = confusion_matrix(yTest, yPred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Voting Classifier")
plt.savefig(f"ConfusionMatrixVotingClassifier{name_dataset}.png")
plt.show()


if tr_score - test_score > 0.1: # if the difference is more than 10%, the model is overfitting
    print(f"the model is overfitting!")

elif test_score < 0.75 and tr_score < 0.75:
    print(f"the model is underfitting!")

elif abs(tr_score-test_score) < 0.05 and test_score > 0.75 and train_score > 0.75:
    print(f"the model is well-fitted!")

else:
    print(f"the model's performance is inconsistent!")

####################### Predicting and Plotting the Confusion Matrix ##########################
# computing the confusion matrix
cm = confusion_matrix(yTest, yPred)#, normalize="true") # with normalize argument: we normalize each row

# plotting using seaborn
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=sorted(set(y)), yticklabels=sorted(set(y)))
#, to set up the format text in the heatmap: fmt="d", "d" stands for integer format (whole number)
# another way of labeling the axis: xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title(f"Confusion Matrix (SVC)")# using the metric {my_grid.scoring}
plt.savefig(f"ConfusionMatrixGridSearch{name_dataset}.png")
plt.show()


# Extract the fitted models from the VotingClassifier
fitted_dt = voting_clf.named_estimators_['dt']
fitted_svc = voting_clf.named_estimators_['svc']
fitted_nb = voting_clf.named_estimators_['nb']
fitted_logreg = voting_clf.named_estimators_['logreg']

# Get predictions from each fitted model
dt_pred = fitted_dt.predict(XTest)
# ChatGPT: suggested to use clip function and keep the outcome within a certain limit.
# but this is not a solution! we are just cutting out what we do not want to see. There is some wrong information is being passed to the model!
dt_pred = np.clip(dt_pred, 3, 8)
svc_pred = fitted_svc.predict(XTest)
svc_pred = np.clip(svc_pred, 3, 8)
nb_pred = fitted_nb.predict(XTest)
nb_pred = np.clip(nb_pred, 3, 8)
logreg_pred = fitted_logreg.predict(XTest)
logreg_pred = np.clip(logreg_pred, 3, 8)

# Convert predictions into a DataFrame for comparison
model_votes = pd.DataFrame({
    'Decision Tree': dt_pred,
    'SVC': svc_pred,
    'Naive Bayes': nb_pred,
    'Logistic Regression': logreg_pred,
    'Final Vote': yPred
})

# Print a sample of votes
print(model_votes.head(10))  # Show first 10 test samples

# I am still debugging why I do have 1 and 2 in my output! therefore I am checking the yTrain, Test, y.
# it gives me the number of each class in a format of a dictionary.
from collections import Counter
print("Class distribution in yTrain:", Counter(yTrain))
print("Class distribution in yTest:", Counter(yTest))
print("Unique class labels in y:", np.unique(y))

# Here I am printing whether there is 1 or 2 in the prediction of each models! the outcome shows there is non! :(
print("Unique class labels predicted by each model:")
print("Decision Tree:", np.unique(dt_pred))
print("SVC:", np.unique(svc_pred))
print("Naive Bayes:", np.unique(nb_pred))
print("Logistic Regression:", np.unique(logreg_pred))
print("Voting Classifier (final prediction):", np.unique(yPred))