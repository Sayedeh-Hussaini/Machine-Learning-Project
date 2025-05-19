import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score, mean_squared_error


name_dataset = "Multicollinearity" #Multicollinearity, DatasetNoDuplicate

# import Data set with no Duplication
dataset = np.loadtxt(f"{name_dataset}.csv", delimiter=",", skiprows=1)

X = dataset[:, :-1]
y = dataset[:, -1]

XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=42, stratify=y)#, stratify=y
print(len(XTest))
# scaling the data
scaler = StandardScaler()
XTrain = scaler.fit_transform(XTrain)
XTest = scaler.transform(XTest)

# Train the model
svc_model = SVC(kernel="rbf", C=0.1, gamma=0.1, class_weight=None, random_state=42) #, class_weight="balanced"
svc_model.fit(XTrain, yTrain)

# Prediction
yPred = svc_model.predict(XTest)

# Evaluate the Model
tr_score = svc_model.score(XTrain, yTrain)
test_score = svc_model.score(XTest, yTest)
balanceaccuracy_score = balanced_accuracy_score(yTest, yPred)
mse = mean_squared_error(yTest, yPred)

print(f"Training Score: {tr_score:.4f}")
print(f"Test Score: {test_score:.4f}")
print(f"Balance Accuracy Score: {balanceaccuracy_score:.4f}")
print(f"mean squared error: {mse:.4f}")

if tr_score - test_score > 0.1: # if the difference is more than 10%, the model is overfitting
    print(f"the model is overfitting!")

elif test_score < 0.75 and tr_score < 0.75:
    print(f"the model is underfitting!")

elif abs(tr_score-test_score) < 0.05 and test_score > 0.75 and tr_score > 0.75:
    print(f"the model is well-fitted!")

else:
    print(f"the model's performance is inconsistent!")



# Classification Report: we see all detailed precision, recall, and F1-score
print("\nClassification Report:")
print(classification_report(yTest, yPred))


###########################Confusion Matrix###############################
conf_matrix = confusion_matrix(yTest, yPred) #, normalize="true"
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"ConfusionMatrixSVM{name_dataset}.png")
plt.show()


############################Comparing the predicted and true values#############################################
plt.scatter(yTest, yPred, alpha=0.6, color="blue", label="predictions")
plt.plot([min(yTest), max(yTest)], [min(yTest), max(yTest)], color="red", linestyle="--", label="Perfect Prediction")
plt.xlabel("Actual Labels (yTest)")
plt.ylabel("Predicted Labels (yPred)")
plt.title("Actual vs. Predicted Labels")
plt.legend()
plt.savefig(f"ScatterPlotTruePred {name_dataset}.png")
plt.show()
