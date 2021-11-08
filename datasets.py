#to download the dataset follow Readme.md file

import matplotlib.pyplot as plt
import pandas as pd
import IPython.display as ipd
import librosa
import librosa.display
from sklearn.preprocessing import LabelEncoder
import tqdm
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

labelencoder=LabelEncoder()

def read_one_file(location):
  data, sample_rate = librosa.load(location, res_type = 'kaiser_fast')
  mf_ccs_features = librosa.feature.mfcc(y = data, sr = sample_rate, n_mfcc = 40)
  mfcc_scaled = np.mean(mf_ccs_features.T, axis = 0)
  return mfcc_scaled

def get_dataset(location):
  data = []
  print('Preparing the dataset!')
  meta_data = pd.read_csv(location + '/metadata/UrbanSound8K.csv')
  for index, row in tqdm(meta_data.iterrows()):
    file_location = location + '/audio/fold' + str(row['fold']) + '/' + row['slice_file_name']
    librosa_data = read_one_file(file_location)
    class_name = row['class']
    data.append([librosa_data, class_name])

  features_df = pd.DataFrame(data, columns=['features', 'class_name'])
  X = np.array(features_df['features'].tolist())
  y = np.array(features_df['class_name'].tolist())
  
  y = to_categorical(labelencoder.fit_transform(y))
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
  return X_train, X_test, y_train, y_test

def get_encoder():
  data = []
  for index, row in tqdm(meta_data.iterrows()):
    file_location = location + '/audio/fold' + str(row['fold']) + '/' + row['slice_file_name']
    librosa_data = read_one_file(file_location)
    class_name = row['class']
    data.append([librosa_data, class_name])
    break;

  features_df = pd.DataFrame(data, columns=['features', 'class_name'])
  X = np.array(features_df['features'].tolist())
  y = np.array(features_df['class_name'].tolist())
  labelencoder=LabelEncoder()
  y = to_categorical(labelencoder.fit_transform(y))
  return labelencoder