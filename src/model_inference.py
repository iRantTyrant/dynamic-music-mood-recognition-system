#We import the tensorflow library 
import tensorflow as tf
#We import the essentia library
import essentia.standard as es
#we import numpy as np

#Now we load the VGGish pretrained model
def load_model(model_path):
	model = tf.saved_model.load(model_path)
	return model

#We use the model to predict the Arousal / Valance by the characteristics we extracted 
def predict_arousal_valence(model, features):
	features = np.expand_dims(features, axis=0)
	prediction = model(features)
	return prediction

