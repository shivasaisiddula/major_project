import pandas as pd
import numpy as np
df=pd.read_csv(r"C:\Users\Shiva Sai\Downloads\data (1).csv")
df.head()

print("General info about colums,rows etc.")
df.info()
print("\nTarget variables value counts\n",df["y"].value_counts())

import matplotlib.pyplot as plt
def hist(df,plt):
  plt.hist(df[df["y"]==1]["y"],label="epileptic seizure activity")
  plt.hist(df[df["y"]!=1]["y"],label="not a seizure")
  plt.legend(loc='lower right')
  plt.show()

hist(df,plt)

df.head(2) #just a quick look to data again. x1...x178 columns are a part of our time-series data (EEG Signals) but Unnamed: 0 column must be inspected.

df["Unnamed: 0"].value_counts #As you can see this column is exlusive for all instance
                              #So it means, the column has no effect on classification, it is unnecessary

df["y"].value_counts() # I will transform 2,3,4,5 classes to 0, 1 class to 1

#This method drop the unnecessary column (Unnamed: 0) and transform the target variable
def prepareData(df):
  df["y"]=[1 if df["y"][i]==1 else 0 for i in range(len(df["y"]))]
  target=df["y"]
  df_copy=df.drop(["Unnamed: 0","y"],axis=1)
  return df_copy,target

df_copy,target=prepareData(df)


import pywt #importing pywt for getting wavelet transform features
from hurst import compute_Hc

def getHurst(df_copy):
  df_copy["hurst_ex"]=[compute_Hc(df_copy.iloc[i], kind="change", simplified=True)[0] for i in range(len(df_copy))]
  df_copy["hurst_c"]=[compute_Hc(df_copy.iloc[i], kind="change", simplified=True)[1] for i in range(len(df_copy))]
  return df_copy


def getStatsForHurst(df_copy):
  plt.scatter(df_copy["hurst_ex"],target)
  print("mean value of hurst exponent for class 1:",np.mean(df_copy.iloc[target[target==1].index]["hurst_ex"]))
  print("mean value of hurst exponent for class 0:",np.mean(df_copy.iloc[target[target==0].index]["hurst_ex"]))
  print("mean value of hurst constant for class 1:",np.mean(df_copy.iloc[target[target==1].index]["hurst_c"]))
  print("mean value of hurst constant for class 0:",np.mean(df_copy.iloc[target[target==0].index]["hurst_c"]))
  print("median value of hurst exponent for class 1:",np.median(df_copy.iloc[target[target==1].index]["hurst_ex"]))
  print("median value of hurst exponent for class 0:",np.median(df_copy.iloc[target[target==0].index]["hurst_ex"]))
  print("median value of hurst constant for class 1:",np.median(df_copy.iloc[target[target==1].index]["hurst_c"]))
  print("median value of hurst constant for class 0:",np.median(df_copy.iloc[target[target==0].index]["hurst_c"]))

#These methods create a new dataset with wavelet transform
#In getWaveletFeatures method, i get a group of wavelet coeffient and hurst exponent and the constant for all instance
#give these values to statisticsForWavelet function to get coeffients quartiles,mean,median,standart deviation,variance,root mean square and some other values.
#Lastly createDfWavelet method give all these values and return a new dataframe
def getWaveletFeatures(data,target):
    list_features = []
    for signal in range(len(data)):
        list_coeff = pywt.wavedec(data.iloc[signal], "db4")
        features = []
        features.append(data.iloc[signal]["hurst_ex"])
        features.append(data.iloc[signal]["hurst_c"])
        for coeff in list_coeff:
            features += statisticsForWavelet(coeff)
        list_features.append(features)
    return createDfWavelet(list_features,target)
#This method taken from [9]
def statisticsForWavelet(coefs):
    n5 = np.nanpercentile(coefs, 5)
    n25 = np.nanpercentile(coefs, 25)
    n75 = np.nanpercentile(coefs, 75)
    n95 = np.nanpercentile(coefs, 95)
    median = np.nanpercentile(coefs, 50)
    mean = np.nanmean(coefs)
    std = np.nanstd(coefs)
    var = np.nanvar(coefs)
    rms = np.nanmean(np.sqrt(coefs**2))
    return [n5, n25, n75, n95, median, mean, std, var, rms]

def createDfWavelet(data,target):
  for i in range(len(data)):
    data[i].append(target[i])
  return pd.DataFrame(data)

df_copy=getHurst(df_copy)
getStatsForHurst(df_copy)

df_copy_fea=getWaveletFeatures(df_copy,target)

df_copy_fea.head()#our new dataset is ready



from sklearn.utils import shuffle
def createBalancedDataset(data,random_state):
  #shuffling for random sampling
  X = shuffle(data,random_state=random_state)
  #getting first 6500 value
  return X.sort_values(by=47, ascending=False).iloc[:6500].index

v=createBalancedDataset(df_copy_fea,42)

plt.hist((df_copy_fea.iloc[v])[47])
(df_copy_fea.iloc[v][47]).value_counts() #more balanced dataset

#normalizing dataset
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_copy_fea.drop([47],axis=1))
n_df_fea=pd.DataFrame(scaler.transform(df_copy_fea.drop([47],axis=1)))






#features are given as input to SVM model

from sklearn.model_selection import train_test_split
X_trainr, X_testr, y_trainr, y_testr = train_test_split(n_df_fea.iloc[v], target.iloc[v], test_size=0.33, random_state=42)


# Check if data is linearly seperable or not
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

lda = LinearDiscriminantAnalysis()
lda.fit(X_trainr, y_trainr) # Train LDA on EEG features
y_lda_pred = lda.predict(X_testr) # Predict on test data

lda_acc = accuracy_score(y_testr, y_lda_pred) # Compute accuracy
print(f"LDA Accuracy: {lda_acc:.2f}")



#Reference point for svm
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

#I will explain this model in model part in the notebook
clf = svm.SVC(kernel="linear", probability=True)
clf.fit(X_trainr, y_trainr)
#cross validation is 10
y_pred = cross_val_predict(clf,X_testr,y_testr,cv=10)
print("All features from SVM are included\n",classification_report(y_testr, y_pred, digits=6))






"""**Selecting most important 20 features with Anova**"""

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif


#Selection most important 20 feature by using Anova test
def selectFeature(X_trainr, y_trainr, X_testr):
    sel_f = SelectKBest(f_classif, k=20)
    X_train_f = sel_f.fit_transform(X_trainr, y_trainr)
    mySelectedFeatures = [i for i in range(len(sel_f.get_support())) if sel_f.get_support()[i] == True]
    
    j = 0
    unseable_columns = []
    for i in X_trainr.columns:
        if j not in mySelectedFeatures:
            unseable_columns.append(i)
        j += 1
    
    X_train_arranged = X_trainr.drop(columns=unseable_columns)
    X_test_arranged = X_testr.drop(columns=unseable_columns)
    
    return X_train_arranged, X_test_arranged, mySelectedFeatures  # Now returning selected features

# Call the function and store the selected features
X_train_arranged, X_test_arranged, mySelectedFeatures = selectFeature(X_trainr, y_trainr, X_testr)


X_train_arranged, X_test_arranged, mySelectedFeatures = selectFeature(X_trainr, y_trainr, X_testr)

X_train_arranged.columns #The most important columns according to Anova



import seaborn as sns
import matplotlib.pyplot as plt

# Get feature scores from SelectKBest
anova_scores = f_classif(X_trainr, y_trainr)[0]  # Extract F-scores

# Store feature names and their corresponding scores
feature_scores = pd.DataFrame({
    "Feature": X_trainr.columns,
    "ANOVA F-Score": anova_scores
})

# Select only the top 20 features
top_20_features = feature_scores.loc[mySelectedFeatures].sort_values(by="ANOVA F-Score", ascending=False)

# Plot bar chart
plt.figure(figsize=(12, 6))  # Increase figure size for better readability
sns.barplot(x=top_20_features.index, y=top_20_features["ANOVA F-Score"], palette="viridis")

plt.xticks(rotation=90)  # Rotate x-axis labels for better visibility
plt.xlabel("Feature Names")
plt.ylabel("ANOVA F-Score (Feature Importance)")
plt.title("Top 20 Features Selected by ANOVA")

plt.show()




#Overall accuracy is decreased
from sklearn import svm
from sklearn.metrics import classification_report
clf = svm.SVC(kernel="linear", probability=True)
clf.fit(X_train_arranged, y_trainr)
y_pred = cross_val_predict(clf,X_test_arranged,y_testr,cv=10)
print("Only Anova test's Features from SVM are used\n",classification_report(y_testr, y_pred, digits=6))








from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce to 2D using PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_trainr)
X_test_pca = pca.transform(X_testr)

# Train SVM on PCA-transformed data
clf_pca = svm.SVC(kernel="linear", probability=True)
clf_pca.fit(X_train_pca, y_trainr)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1, 100),
                     np.linspace(X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1, 100))

Z = clf_pca.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap=plt.cm.coolwarm, alpha=0.6)
plt.contour(xx, yy, Z, colors='k', levels=[0], linewidths=2, label="Hyperplane")  # Hyperplane

# Scatter plot for actual points
seizure_mask = (y_trainr == 1)
non_seizure_mask = (y_trainr == 0)

plt.scatter(X_train_pca[non_seizure_mask, 0], X_train_pca[non_seizure_mask, 1], 
            color='blue', edgecolors='k', label="Non-Seizures (Blue Dots)")
plt.scatter(X_train_pca[seizure_mask, 0], X_train_pca[seizure_mask, 1], 
            color='red', edgecolors='k', label="Seizures (Red Dots)")

# Highlight support vectors
plt.scatter(clf_pca.support_vectors_[:, 0], clf_pca.support_vectors_[:, 1], 
            s=60, facecolors='none', edgecolors='white', label="Support Vectors")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Decision Boundary (All Features)")
plt.legend()
plt.show()








# Apply PCA on the selected top 20 features
X_train_pca_anova = pca.fit_transform(X_train_arranged)
X_test_pca_anova = pca.transform(X_test_arranged)

# Train SVM on PCA-transformed ANOVA features
clf_pca_anova = svm.SVC(kernel="linear", probability=True)
clf_pca_anova.fit(X_train_pca_anova, y_trainr)

# Plot decision boundary
xx, yy = np.meshgrid(np.linspace(X_train_pca_anova[:, 0].min() - 1, X_train_pca_anova[:, 0].max() + 1, 100),
                     np.linspace(X_train_pca_anova[:, 1].min() - 1, X_train_pca_anova[:, 1].max() + 1, 100))

Z = clf_pca_anova.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap=plt.cm.coolwarm, alpha=0.6)
plt.contour(xx, yy, Z, colors='k', levels=[0], linewidths=2, label="Hyperplane")  # Hyperplane

# Scatter plot for actual points
seizure_mask = (y_trainr == 1)
non_seizure_mask = (y_trainr == 0)

plt.scatter(X_train_pca_anova[non_seizure_mask, 0], X_train_pca_anova[non_seizure_mask, 1], 
            color='blue', edgecolors='k', label="Non-Seizures (Blue Dots)")
plt.scatter(X_train_pca_anova[seizure_mask, 0], X_train_pca_anova[seizure_mask, 1], 
            color='red', edgecolors='k', label="Seizures (Red Dots)")

# Highlight support vectors
plt.scatter(clf_pca_anova.support_vectors_[:, 0], clf_pca_anova.support_vectors_[:, 1], 
            s=60, facecolors='none', edgecolors='white', label="Support Vectors")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("SVM Decision Boundary (Top 20 ANOVA Features)")
plt.legend()
plt.show()









import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming 'predictions' contains the model's predicted values or decision function scores
plt.hist(y_pred, bins=50)


# Set x-axis ticks with unit spacing
xmin, xmax = plt.xlim()
plt.xticks(np.arange(np.floor(xmin), np.ceil(xmax) + 1, 1))

# Label the axes
plt.xlabel("Predicted Values / Decision Function Scores")  # X-axis
plt.ylabel("Frequency")  # Y-axis

# Add a title
plt.title("Distribution of Model Predictions")

# Show the plot
plt.show()

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_testr, clf.decision_function(X_test_arranged))
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - SVM')
plt.legend(loc="lower right")
plt.show()

















import time
import memory_profiler
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, f_classif


# Train-test split
X_trainr, X_testr, y_trainr, y_testr = train_test_split(n_df_fea.iloc[v], target.iloc[v], test_size=0.33, random_state=42)

# Feature selection (Using ANOVA F-score)
selector = SelectKBest(score_func=f_classif, k=20)
selector.fit(X_trainr, y_trainr)
selected_features = X_trainr.columns[selector.get_support()]  # Get top 20 feature names

# Create feature-selected datasets
X_train_selected = X_trainr[selected_features]
X_test_selected = X_testr[selected_features]

# Function to measure training time and memory usage
def train_model_with_metrics(model, X_train, y_train, X_test, y_test, model_name):
    print(f"\n=== Training {model_name} ===")
    
    # Start timing and memory profiling
    start_time = time.time()
    mem_usage = memory_profiler.memory_usage((model.fit, (X_train, y_train)), max_usage=True)
    training_time = time.time() - start_time

    # Cross-validation prediction
    y_pred = cross_val_predict(model, X_test, y_test, cv=10)
    
    # Print results
    print(f"{model_name} Training Time: {training_time:.3f} seconds")
    print(f"{model_name} Peak Memory Usage: {mem_usage:.3f} MB")
    

# 1️**SVM with All Features**
svm_all_features = svm.SVC(kernel="linear", probability=True)
train_model_with_metrics(svm_all_features, X_trainr, y_trainr, X_testr, y_testr, "SVM with All Features")

# 2️**SVM with Selected Features**
svm_selected_features = svm.SVC(kernel="linear", probability=True)
train_model_with_metrics(svm_selected_features, X_train_selected, y_trainr, X_test_selected, y_testr, "SVM with Selected Features")
