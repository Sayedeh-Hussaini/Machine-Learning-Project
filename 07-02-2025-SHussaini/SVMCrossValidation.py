import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, balanced_accuracy_score


name_dataset = "DatasetNoDuplicate" #Multicollinearity, DatasetNoDuplicate
print(f"Dataset is: {name_dataset}")
# import Data set with no Duplication
dataset = np.loadtxt(f"{name_dataset}.csv", delimiter=",", skiprows=1)

X = dataset[:, :-1]
y = dataset[:, -1]

# here we use stratify argument because we want to have a representative distribution of all the classes in our training and test data
XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)#, stratify=y

# instance of our object
scaler = StandardScaler()
# here we fit and transform our training data
XTrain = scaler.fit_transform(XTrain)

# here we just transform the data set to avoid information leakage
XTest = scaler.transform(XTest)

# Train the model
svc_model = SVC(kernel="rbf", C=0.1, gamma=0.1, class_weight=None, random_state=42) #
svc_model.fit(XTrain, yTrain)

# Prediction
yPred = svc_model.predict(XTest)

##########Stratify Each Fold during Cross Validation in GridSearchCV####################
StratifiedData = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
########################################################################################

# Cross validation with five folds
scores = cross_val_score(svc_model, XTrain, yTrain, cv=StratifiedData)

# Mean CV score
cvmean_score = np.mean(scores)

# Compute the Training Score
tr_score = svc_model.score(XTrain, yTrain)
# Compute Test Score
test_score = svc_model.score(XTest, yTest)

balanceaccuracy_score = balanced_accuracy_score(yTest, yPred)
mse = mean_squared_error(yTest, yPred)

# printing out information about the scores
print(f"Cross Validation Score: {cvmean_score:.4f}")
print(f"Standard Deviation of the CV Scores: {np.std(cvmean_score)}")
print(f"Training Score: {tr_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Balance Accuracy Score: {balanceaccuracy_score:.4f}")
print(f"mean squared error: {mse:.4f}")


if cvmean_score - test_score > 0.1: # if the difference is more than 10%, the model is overfitting
    print(f"the model is overfitting!")

elif test_score < 0.75 and cvmean_score < 0.75:
    print(f"the model is underfitting!")

elif abs(cvmean_score-test_score) < 0.05 and test_score > 0.75 and cvmean_score > 0.75:
    print(f"the model is well-fitted!")

else:
    print(f"the model's performance is inconsistent!")


# Classification Report: we see all detailed precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(yTest, yPred))

