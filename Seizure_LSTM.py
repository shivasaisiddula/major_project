import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

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




from sklearn.metrics import classification_report

# Split dataset
X_trainr, X_testr, y_trainr, y_testr = train_test_split(n_df_fea, target, test_size=0.33, random_state=42)

def add_noise(data, noise_level=0.05):
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise

# Add noise to training data for augmentation
X_trainr_noisy = add_noise(X_trainr)
X_trainr_augmented = np.vstack((X_trainr, X_trainr_noisy))
y_trainr_augmented = np.hstack((y_trainr, y_trainr))

# Reshape for LSTM
X_trainrr = np.array(X_trainr_augmented).reshape(X_trainr_augmented.shape[0], X_trainr_augmented.shape[1], 1)
X_testrr = np.array(X_testr).reshape(X_testr.shape[0], X_testr.shape[1], 1)

# Build LSTM Model
model = Sequential()
model.add(Input(shape=(X_trainrr.shape[1], X_trainrr.shape[2])))  # Correct Input Shape
model.add(LSTM(50, return_sequences=True))  # First LSTM Layer
model.add(Dropout(0.3))  # Increased Dropout
model.add(LSTM(25))  # Second LSTM Layer
model.add(Dropout(0.3))  # Another Dropout
model.add(Dense(1, activation='sigmoid'))

# Compile Model with Adam Optimizer & Lower Learning Rate
model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train Model
history = model.fit(X_trainrr, y_trainr_augmented, epochs=50, batch_size=32, validation_data=(X_testrr, y_testr), verbose=2, shuffle=False, callbacks=[early_stopping])


# Predict on test data
y_pred_prob = model.predict(X_testrr)  # Get probability scores
y_pred = (y_pred_prob > 0.5).astype(int)  # Convert probabilities to binary predictions

# Generate classification report
report = classification_report(y_testr, y_pred, digits=6)
print(report)





from tensorflow.keras.utils import plot_model

# Save model architecture as an image
plot_model(model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)







# Show the plot
plt.show()
plt.xlabel("Predicted Values / Decision Function Scores")  # X-axis
plt.ylabel("Frequency")  # Y-axis

# Add a title
plt.title("Distribution of Model Predictions")
# Plot Training and Validation Loss
# Plot Training and Validation Loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.xlabel("Epochs")  # X-axis label
plt.ylabel("Loss")  # Y-axis label
plt.title("Training and Validation Loss")  # Plot title
plt.legend()
plt.show()






from sklearn.metrics import roc_curve, auc

# Compute ROC curve and AUC
fpr, tpr, _ = roc_curve(y_testr, y_pred_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Diagonal reference line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()





import time
import memory_profiler

# Function to measure training time & memory usage
def train_lstm_with_metrics(model, X_train, y_train, X_test, y_test, batch_size=32, epochs=50):
    start_time = time.time()  # Start timer

    # Measure memory usage during training
    mem_usage = memory_profiler.memory_usage(
        (model.fit, (X_train, y_train), {'epochs': epochs, 'batch_size': batch_size, 
                                         'validation_data': (X_test, y_test), 
                                         'verbose': 2, 'shuffle': False, 
                                         'callbacks': [early_stopping]}),
        max_usage=True
    )

    training_time = time.time() - start_time  # Calculate training time

    print(f"\nLSTM Training Time: {training_time:.3f} seconds")
    print(f"LSTM Peak Memory Usage: {mem_usage:.3f} MiB")

# Train the LSTM model while tracking metrics
train_lstm_with_metrics(model, X_trainrr, y_trainr_augmented, X_testrr, y_testr)
