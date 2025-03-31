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




from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.metrics import classification_report



# Train-test split
X_trainr, X_testr, y_trainr, y_testr = train_test_split(n_df_fea.iloc[v], target.iloc[v], test_size=0.33, random_state=42)

# Check dataset shapes
print("Training Data Shape:", X_trainr.shape, "Testing Data Shape:", X_testr.shape)

# Random Forest Grid Search
grids = {
    'max_depth': [2, 5],
    'min_samples_split': [2, 5, 10],
    'n_estimators': [100, 150],
    'max_features': ['sqrt', 'log2']
}

rfc = RandomForestClassifier(n_jobs=-1, max_features='sqrt', n_estimators=50, oob_score=True, random_state=42)
CV_rfc = GridSearchCV(estimator=rfc, param_grid=grids, cv=5)
CV_rfc.fit(X_trainr, y_trainr)
print("Best Hyperparameters:", CV_rfc.best_params_)

# Train Random Forest with best parameters
clf = RandomForestClassifier(random_state=42, **CV_rfc.best_params_)
clf.fit(X_trainr, y_trainr)

# Feature importance
zipped = pd.DataFrame(zip(X_trainr.columns, clf.feature_importances_), columns=["column", "importance"])
zipped = zipped.sort_values(by="importance", ascending=False)
print("Top 20 Features Selected:", zipped.head(20))

import seaborn as sns
import matplotlib.pyplot as plt

# Select top 20 features
top_20_features = zipped.head(20)

# Increase figure size
plt.figure(figsize=(14, 6))

# Vertical bar plot
sns.barplot(x=top_20_features["column"], y=top_20_features["importance"], palette="viridis")

# Labels & Title
plt.xlabel("Features", fontsize=12)
plt.ylabel("Feature Importance", fontsize=12)
plt.title("Top 20 Most Important Features in EEG Seizure Detection", fontsize=14)

# Rotate x-axis labels
plt.xticks(rotation=90, fontsize=10)

# Adjust layout to prevent cutting off labels
plt.tight_layout()

# Show Plot
plt.show()

# Check selected features in train and test sets
print("Training Features:", X_trainr.columns)
print("Testing Features:", X_testr.columns)

# Validate feature consistency
selected_features = zipped.iloc[:20].index
X_train_selected = X_trainr[selected_features]
X_test_selected = X_testr[selected_features]

# Cross-validation check
scores = cross_val_score(clf, X_testr, y_testr, cv=10)
print("Cross-validation scores:", scores)

# Evaluate Random Forest with all features
y_pred2 = cross_val_predict(clf, X_testr, y_testr, cv=10)
print("Random Forest (All Features)\n", classification_report(y_testr, y_pred2, digits=6))







# Single Decision tree visualization
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))
plot_tree(clf.estimators_[0], filled=True, feature_names=X_trainr.columns, class_names=["Non-Seizure", "Seizure"])
plt.show()

from sklearn.tree import export_text
tree_rules = export_text(clf.estimators_[0], feature_names=X_trainr.columns.tolist())
print(tree_rules)







# Evaluate SVM with selected features
clf_svm = svm.SVC(kernel="linear", probability=True)
clf_svm.fit(X_train_selected, y_trainr)
y_pred_svm = cross_val_predict(clf_svm, X_test_selected, y_testr, cv=10)
print("SVM (Selected Features)\n", classification_report(y_testr, y_pred_svm, digits=6))

# Evaluate Random Forest with selected features
y_pred_sel = cross_val_predict(clf, X_test_selected, y_testr, cv=10)
print("Random Forest (top selected Features)\n", classification_report(y_testr, y_pred_sel, digits=6))


#Random forest, I got hyperparameters from above grid-search
clf1 = RandomForestClassifier(random_state=42, max_depth=5, max_features='sqrt', min_samples_split=5, n_estimators=150)
clf1.fit(X_trainr, y_trainr)
y_pred = cross_val_predict(clf1,X_testr,y_testr,cv=10)
print("Random Forest hypertuned featuares \n",classification_report(y_testr, y_pred, digits = 6))





import seaborn as sns
import matplotlib.pyplot as plt

# Sample accuracy scores (Replace with actual grid search results)
max_depth_values = [5, 10, 15]
n_estimators_values = [50, 100, 200]
min_samples_split_values = [2, 5]  # Two different heatmaps
max_features_values = ['sqrt', 'log2']  # Color differentiation

# Random accuracy values for demonstration
np.random.seed(42)
accuracy_data = {
    (min_samples_split, max_features): np.round(np.random.uniform(0.91, 0.93, (3,3)), 4)
    for min_samples_split in min_samples_split_values
    for max_features in max_features_values
}

# Plotting heatmaps for different min_samples_split values
fig, axes = plt.subplots(1, len(min_samples_split_values), figsize=(12, 5))

for i, min_samples_split in enumerate(min_samples_split_values):
    ax = axes[i]
    for max_features in max_features_values:
        data = accuracy_data[(min_samples_split, max_features)]
        sns.heatmap(data, annot=True, fmt=".4f", cmap="coolwarm", xticklabels=n_estimators_values,
                    yticklabels=max_depth_values, ax=ax, cbar=True)
        ax.set_title(f"min_samples_split={min_samples_split}, max_features={max_features}")
        ax.set_xlabel("Number of Estimators")
        ax.set_ylabel("Max Depth")

plt.suptitle("Hyperparameter Tuning Accuracy Heatmaps")
plt.tight_layout()
plt.show()







from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probability scores for ROC Curve
y_score_rf = clf.predict_proba(X_testr)[:, 1]  # Random Forest
y_score_svm = clf_svm.decision_function(X_test_selected)  # SVM


# Label the axes
plt.xlabel("Predicted Values / Decision Function Scores")  # X-axis
plt.ylabel("Frequency")  # Y-axis

# Add a title
plt.title("Distribution of Model Predictions")


# Compute ROC curve and AUC for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_testr, y_score_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# Compute ROC curve and AUC for SVM
fpr_svm, tpr_svm, _ = roc_curve(y_testr, y_score_svm)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC Curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='blue', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.3f})')
plt.plot(fpr_svm, tpr_svm, color='red', lw=2, label=f'SVM (AUC = {roc_auc_svm:.3f})')

# Random Classifier Line
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random Classifier')

# Labels & Legend
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for EEG Seizure Detection')
plt.legend(loc='lower right')
plt.grid()

# Show Plot
plt.show()




import time
import memory_profiler
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import classification_report

# Train-test split
X_trainr, X_testr, y_trainr, y_testr = train_test_split(n_df_fea.iloc[v], target.iloc[v], test_size=0.33, random_state=42)

# Feature selection
selected_features = zipped.iloc[:20].index
X_train_selected = X_trainr[selected_features]
X_test_selected = X_testr[selected_features]

# Function to measure training time and memory usage
def train_model_with_metrics(model, X_train, y_train, model_name):
    start_time = time.time()
    mem_usage = memory_profiler.memory_usage((model.fit, (X_train, y_train)), max_usage=True)
    training_time = time.time() - start_time
    print(f"\n{model_name} Training Time: {training_time:.3f} seconds")
    print(f"{model_name} Peak Memory Usage: {mem_usage:.3f} MB")

   
# 1️**Random Forest (All Features)**
rf_all = RandomForestClassifier(random_state=42, **CV_rfc.best_params_)
train_model_with_metrics(rf_all, X_trainr, y_trainr, "Random Forest (All Features)")

# 2️**Random Forest (Top Selected Features)**
rf_top_features = RandomForestClassifier(random_state=42, **CV_rfc.best_params_)
train_model_with_metrics(rf_top_features, X_train_selected, y_trainr, "Random Forest (Top Selected Features)")

# 3️**Random Forest (Hypertuned Features)**
rf_hyper = RandomForestClassifier(random_state=42, max_depth=5, max_features='sqrt', min_samples_split=5, n_estimators=150)
train_model_with_metrics(rf_hyper, X_trainr, y_trainr, "Random Forest (Hypertuned Features)")

# 4️**SVM (Selected Features)**
svm_model = svm.SVC(kernel="linear", probability=True)
train_model_with_metrics(svm_model, X_train_selected, y_trainr, "SVM (Selected Features)")
