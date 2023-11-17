import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="imbalanced-learn")

# supervised example.
# cols
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv("magic04.data", names=cols)
print(df.head())
#  g and h to 1 and 0s
df["class"] = (df["class"] == "g").astype(int)
print(df.head())
# ---------------------
for label in cols[:-1]:
  # normalising distributions: 200 of type 1 and 50 of type 2 : histograms would be hard to compare - 1 bigger than the other.
  # so distributing over how many samples are there.
  plt.hist(df[df["class"]==1][label], color='blue', label='gamma', alpha=0.7, density=True)
  plt.hist(df[df["class"]==0][label], color='red', label='hadron', alpha=0.7, density=True)
  plt.title(label)
  plt.ylabel("Probability")
  plt.xlabel(label)
  plt.legend()
  # plt.show()
# train validation and test datasets
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
# some column may have value in 100s other in 0.1 range. may give wrong results.
print(len(train[train["class"] == 1]))
print(len(train[train["class"] == 0]))
def scale_dataset(dataframe, oversample=False):
  X = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values
  scaler = StandardScaler()
  X = scaler.fit_transform(X)
  if oversample:
    print("over")
    ros = RandomOverSampler()
    # take from less class and keep sampling more from the same.
    X, y = ros.fit_resample(X, y)
  # whole data as 2d item
  data = np.hstack((X, np.reshape(y, (-1, 1))))
  return data, X, y
train, X_train, y_train = scale_dataset(train, oversample=True)
print(len(y_train),sum(y_train==1),sum(y_train==0))
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
print(len(y_valid),sum(y_valid==1),sum(y_valid==0))
test, X_test, y_test = scale_dataset(test, oversample=False)
print(len(y_test),sum(y_test==1),sum(y_test==0))

