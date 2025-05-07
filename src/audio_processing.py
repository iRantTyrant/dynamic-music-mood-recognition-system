#We import the essentia library 
import essentia.standard as es
 
#We import the numpy library
import numpy as np


#Function so we can load the music files (We will use essentia's audio loader)
def load_audio(file_path):
	audio = es.MonoLoader(filename = file_path)()
	return audio 


#Function to extract the characteristics of the music (we can use MFCC or a MEL spectogram)
def extract_features(audio):
	#We will use a MEL spectogram in our project
	
	#We create the spectrogram
	spectrum = es.Spectrum()
	melbands = es.MelBands(numberBands=40, lowFrequencyBound=50,highFrequencyBound=8000)
	
	#We calculate the MEL spectrogram 
	mel_spectrum = [melbands(spectrum(frame)) for frame in es.FrameGenerator(audio, frameSize=2048, hopSize=512)]
	mel_spectrum = np.array(mel_spectrum)
	
	return mel_spectrum
	
def pad_features(features, target_shape=(21865, 128)):
    padded_features = np.pad(features, ((0, 0), (0, target_shape[1] - features.shape[1])), 'constant')
    return padded_features

