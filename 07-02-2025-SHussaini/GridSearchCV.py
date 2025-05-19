import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedShuffleSplit, StratifiedKFold, learning_curve
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, balanced_accuracy_score, mean_squared_error

name_dataset = "DatasetNoDuplicate" #Multicollinearity, DatasetNoDuplicate
print(f"the Dataset is: {name_dataset}")

# import Data set with no Duplication
dataset = np.loadtxt(f"{name_dataset}.csv", delimiter=",", skiprows=1)

X = dataset[:, :-1]
y = dataset[:, -1]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)#, stratify=y


##########Stratify Each Fold during Cross Validation in GridSearchCV####################
StratifiedData = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
########################################################################################


svc_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", SVC(random_state=42)) #class_weight="balanced" class_weight argument handle the imbalanced data set which is true in our case
])


svc_grid_params = {
    "scaler": [StandardScaler(), MinMaxScaler(), RobustScaler()],
    "classifier__kernel": ["poly", "linear", "rbf" ],
    "classifier__degree": [2, 3],
    "classifier__class_weight": ["balanced", None],
    "classifier__C": [0.01, 0.1, 1], #, 10, 100 # I tried to avoid large numbers of our regularization parameter because in our data we have outliers
    "classifier__gamma": ["scale", "auto", 0.001, 0.01] #, 0.1, 1
}

# Creating a gridsearch with different parameters
my_grid = GridSearchCV(svc_pipeline, svc_grid_params, cv=StratifiedData, n_jobs=-1, verbose=1) #, scoring="f1_macro" "precision_macro", "recall_macro", "f1_macro"

# Fitting the grid to the training data
my_grid.fit(XTrain, yTrain)

#############################################################################
# Printing some necessary information about the output of gridsearchcv
print(f"the best mean score ({my_grid.scoring}): {my_grid.best_score_}")
print(f"the best parameters: {my_grid.best_params_}")
print(f"the best estimator: {my_grid.best_estimator_}")
print(f"the best score has index: {my_grid.best_index_}")
print(f"the list of std scores: {np.mean(my_grid.cv_results_["std_test_score"])}")


###############################################################################
# converting the gridsearch result to dataframe format
grid_result = pd.DataFrame(my_grid.cv_results_)
grid_result.sort_values(by="mean_test_score", ascending=False, inplace=True)
grid_result.to_csv("SVCFinalResult.csv")


################### Evaluating the Model ##############################
# using the best estimator from GridSearchCV
best_model = my_grid.best_estimator_

# predicting on the test data set
yPred = best_model.predict(XTest)

train_score = best_model.score(XTrain, yTrain)
test_score = best_model.score(XTest, yTest)
balanceaccuracy_score = balanced_accuracy_score(yTest, yPred)
mse = mean_squared_error(yTest, yPred)

print(f"training accuracy: {train_score}")
print(f"test accuracy: {test_score}")
print(f"balanced accuracy: {balanceaccuracy_score}")
print(f"mean squared error: {mse:.4f}")

# Classification Report: we see all detailed precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(yTest, yPred))

if train_score - test_score > 0.1: # if the difference is more than 10%, the model is overfitting
    print(f"the model is overfitting!")

elif test_score < 0.75 and train_score < 0.75:
    print(f"the model is underfitting!")

elif abs(train_score-test_score) < 0.05 and test_score > 0.75 and train_score > 0.75:
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


####################### Learning Curves ##########################

# Generate learning curves to visualize overfitting or underfitting
train_sizes, train_scores, val_scores = learning_curve(
    best_model, XTrain, yTrain, cv=StratifiedKFold(n_splits=5), n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5) #, scoring="f1_macro"
)

# Calculate mean and std for plot
train_mean = np.mean(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_std = np.std(val_scores, axis=1)

# Plot learning curve
plt.plot(train_sizes, train_mean, label="Training score", color="blue")
plt.plot(train_sizes, val_mean, label="Validation score", color="red")
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color="red")

plt.xlabel("Training Set Size")
plt.ylabel("Accuracy Score") #
plt.title("Learning Curve")
plt.legend(loc="best")
plt.savefig(f"LearningCurve {name_dataset}.png")
plt.show()
