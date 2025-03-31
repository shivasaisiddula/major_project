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









import matplotlib.pyplot as plt

# Data preparation
categories = ["Hurst Exponent", "Hurst Constant"]
metrics = ["Mean", "Median"]
class_0_values = [
    np.mean(df_copy.iloc[target[target==0].index]["hurst_ex"]),
    np.mean(df_copy.iloc[target[target==0].index]["hurst_c"]),
    np.median(df_copy.iloc[target[target==0].index]["hurst_ex"]),
    np.median(df_copy.iloc[target[target==0].index]["hurst_c"])
]
class_1_values = [
    np.mean(df_copy.iloc[target[target==1].index]["hurst_ex"]),
    np.mean(df_copy.iloc[target[target==1].index]["hurst_c"]),
    np.median(df_copy.iloc[target[target==1].index]["hurst_ex"]),
    np.median(df_copy.iloc[target[target==1].index]["hurst_c"])
]

# Setting up positions for grouped bars
x_labels = ["Mean Hurst Exponent", "Mean Hurst Constant", "Median Hurst Exponent", "Median Hurst Constant"]
x = np.arange(len(x_labels))  # Bar positions
width = 0.35  # Width of bars

# Plotting
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, class_0_values, width, label='Class 0', color='blue', alpha=0.7)
bars2 = ax.bar(x + width/2, class_1_values, width, label='Class 1', color='orange', alpha=0.7)

# Adding labels
ax.set_xlabel("Attributes")
ax.set_ylabel("Values")
ax.set_title("Mean and Median of Hurst Exponent & Constant")
ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=20)
ax.legend()

# Add spacing between groups (after every two bars)
plt.xticks(np.arange(len(x_labels)), x_labels)
plt.grid(axis='y', linestyle="--", alpha=0.7)

plt.show()






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Assuming df_copy_fea is your DataFrame and 'class' is the target column
df_class_0 = df_copy_fea[df_copy_fea[47] == 0].drop(47, axis=1)  # Drop target column
df_class_1 = df_copy_fea[df_copy_fea[47] == 1].drop(47, axis=1)

# Compute mean and median
median_class_0 = df_class_0.median()
median_class_1 = df_class_1.median()

mean_class_0 = df_class_0.mean()
mean_class_1 = df_class_1.mean()

# Apply log-scaling (handling zero values)
median_class_0_log = np.log1p(median_class_0 + 1e-6)
median_class_1_log = np.log1p(median_class_1 + 1e-6)

mean_class_0_log = np.log1p(mean_class_0 + 1e-6)
mean_class_1_log = np.log1p(mean_class_1 + 1e-6)

# Plot both mean and median comparisons
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

x_labels = df_class_0.columns.astype(str)
x_positions = np.arange(len(x_labels))  # Numerical positions for bars

# Plot median values
axes[0].bar(x_positions, median_class_0_log, color='blue', alpha=0.6, label='Class 0 (Non-seizure)')
axes[0].bar(x_positions, median_class_1_log, color='red', alpha=0.6, label='Class 1 (Seizure)')
axes[0].set_title("Comparison of Log-Scaled Median Values (Before Normalization)")
axes[0].set_ylabel("Log-Scaled Median Value (log1p)")
axes[0].legend()
axes[0].set_xticks(x_positions)
axes[0].set_xticklabels(x_labels, rotation=90, fontsize=10)  # Ensure numbers are visible

# Plot mean values
axes[1].bar(x_positions, mean_class_0_log, color='blue', alpha=0.6, label='Class 0 (Non-seizure)')
axes[1].bar(x_positions, mean_class_1_log, color='red', alpha=0.6, label='Class 1 (Seizure)')
axes[1].set_title("Comparison of Log-Scaled Mean Values (Before Normalization)")
axes[1].set_ylabel("Log-Scaled Mean Value (log1p)")
axes[1].set_xlabel("Features")
axes[1].legend()
axes[1].set_xticks(x_positions)
axes[1].set_xticklabels(x_labels, rotation=90, fontsize=10)  # Ensure numbers are visible

plt.tight_layout()
plt.show()








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

