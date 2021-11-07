import tensorflow as tf
from model import *
from datasets import *
import argparse

classes = ['Dog bark', 'Jack Hammer', 'Drilling', 'Air Conditioner', 'Engine Idling', 'Children Playing', 'Street Music', 'Siren', 'Car Horn', 'Gun Shot']
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--saved_weights', type=str, default = None, help = 'h5 file that has saved weights from the train.py file')
	parser.add_argument('--music_file', type=str, default = 100, help = 'Path to music file')
	opt = parser.parse_args()

	model = get_model()
	model.load_weights(opt.saved_weights)

	audio, sample_rate = librosa.load(opt.music_file, res_type='kaiser_fast') 
	mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
	mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

	#print(mfccs_scaled_features)
	mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
	#print(mfccs_scaled_features)
	#print(mfccs_scaled_features.shape)
	predicted_label=model.predict(mfccs_scaled_features)
	predicted_label=np.argmax(predicted_label,axis=1)
	print(predicted_label)
	labelencoder = get_encoder()
	prediction_class = labelencoder.inverse_transform(predicted_label) 
	print('The sound of ', classes[prediction_class])